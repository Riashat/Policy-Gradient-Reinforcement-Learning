classdef Rho_Integrator < handle
    
    properties(SetAccess= private)       
        expected_values;          
        rho_start_states;
        rho_start_actions;
        rho_successor_states;
        rho_successor_actions;
        rhoStart;
        mdp;
        actions_count;
        states_count;            
    end
     
    methods (Access=public)        
        function obj = Rho_Integrator (expected_values,mdp)
            obj.expected_values = expected_values;
            obj.mdp = mdp;
            obj.actions_count = 10;
            obj.states_count = 10;            
        end
        
        %function to initialize the successive states and actions
        function SetRho(obj, agent)              
            obj.getStartStatesActions(agent);        
            obj.getSuccessiveStatesActions(agent);
            obj.Init(obj.mdp.H);
                  
        end
        
              
        function integralResult = Integration(obj, Product_Function)           
            integralResult = squeeze( sum( bsxfun(@times , full(obj.rhoStart),Product_Function(obj.rho_start_states, obj.rho_start_actions)))) + ...
                squeeze( sum(Product_Function(obj.rho_successor_states, obj.rho_successor_actions)));
            integralResult = full(integralResult);

        end
        
       

    end
    
    methods (Access = private)
        
        function Init(obj, H)
            
            gamma = obj.mdp.gamma;
            
            xiStart = obj.getXiStart();
            obj.rhoStart = xiStart*gamma^0;
            
        end
        
        
        
        
        function getSuccessiveStatesActions(obj, agent)
            new_traj_count = size(obj.expected_values.new_TRAJ, 1);           
            obj.rho_successor_states = nan(new_traj_count*obj.actions_count, obj.mdp.state_dim);
            obj.rho_successor_actions = nan(new_traj_count*obj.actions_count, obj.mdp.action_dim);
                                
            for i = 1:new_traj_count
                successorState = obj.expected_values.new_TRAJ(i,:);
                for j = 1:obj.actions_count
                    idx = sub2ind([new_traj_count obj.actions_count], i, j );                   
                    obj.rho_successor_states(idx,:)  = successorState;
                   
                    %using a deterministic policy 
                    obj.rho_successor_actions(idx,:) = agent.policy_exploratory(successorState);
                end
            end
        end
    
               
        function getStartStatesActions(obj, agent)         
            obj.rho_start_states = nan(obj.states_count*obj.actions_count, obj.mdp.state_dim);
            obj.rho_start_actions = nan(obj.states_count*obj.actions_count, obj.mdp.action_dim);            
            for i = 1:obj.states_count
                startState = obj.mdp.getStartState();
                for j = 1:obj.actions_count
                    idx = sub2ind([obj.states_count obj.actions_count], i, j );                    
                    obj.rho_start_states(idx,:) = startState;                    
                    obj.rho_start_actions(idx,:) = agent.policy_exploratory(startState);                  
                end
            end
        end
                 
        function xiStart = getXiStart(obj)
            count = obj.states_count * obj.actions_count;
            xiStart = repmat(1/count, [count 1]);
        end
        
        
    end


end



            
        
    
    