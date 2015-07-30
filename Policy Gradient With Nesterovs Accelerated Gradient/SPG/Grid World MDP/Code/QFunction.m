classdef QFunction
    
 
    %this class has nothing defined for Q function approximation
    %how to approximate the Q function or use value function or
    %compatible function approximation?
    
    
    properties (SetAccess = private)
        expected_values;
        mdp;
        Value;
             
    end
    
    
        methods(Access=public)
        
        function obj = QFunction (expected_values,mdp)        
            obj.mdp = mdp;
            obj.expected_values = expected_values;
                        
        end
        
        %for the Q function
        
        %need the V function 
        % reward + gamma * V function
        
        
        
        %function to compute the Value Functions
        function value = ValueFunction (obj, agent, old_V)
            
            actions = agent.policy(obj.expected_values.new_TRAJ);
            updated_actions = [obj.expected_values.new_TRAJ actions];
            value_reward = obj.mdp.reward (obj.expected_values.new_TRAJ, actions);
            
            loop = 2500;
            j=0;
            iter =1;
           
            if (exist('old_V', 'var'))
                value = old_V;
            else
                    
            value = value_reward;
            end
            
            
            
            while((iter>0.000000001) && (j<loop))
                     
                j = j+1;
                prev_value = value;
                
                %for value iteration, the next state value is conditional
                %on the previous state value
                %this is not done here - or how to do it?
               % value = value_reward + obj.mdp.gamma * obj.value % * P
                %transition dynamics
                
                
                %do it for infinite - discretize a continuous mdp
                % not applicable - only for finite horizon, but MDP is
                % infinite horizon
                value = value_reward + obj.mdp.gamma * obj.expected_values.conditional_expectation(value);
                iter = sum(abs(value - old_V));             
            end
            
            obj.Value = value;
                       
        end
               
        function Q_Val = Q_Value (obj, states, actions)
           Q_Val = obj.mdp.reward(states, actions) + obj.mdp.gamma * obj.expected_values.conditional_expectation(obj.Value);

        end
   
               
        end
        
        
end


        
            
            