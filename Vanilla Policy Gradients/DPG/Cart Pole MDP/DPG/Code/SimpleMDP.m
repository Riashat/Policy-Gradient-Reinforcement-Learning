classdef SimpleMDP < MDP & DistanceMeasure
    
    properties(SetAccess = private)
        
    end
    
    methods(Access = public)
        
        function obj = SimpleMDP(noise, gamma, H, actionDiscCount)
            
            noise=0;
            gamma=0.99;
            H=100;
            Actions = 50;
            
            start_state_noise = 0.1 ;
            state_dim = 1;
            action_dim = 1;
            max_action = 1;
            %for discretizing
            min_disc_act = -1;
            max_disc_act = 1;
            
            %for sampling
            max_state = 4;
            min_state = -4;
            name = 'SimpleMDP';
            
            actionDiscCount = 10;
            obj = obj@MDP(gamma, H, noise, start_state_noise, state_dim, action_dim, max_action,...
                min_disc_act, max_disc_act, max_state, min_state, actionDiscCount, name);
            obj.discAct = 0; %continuous
   
        end
        
      
        function [distsqr] = sqDistance(obj, X, Z)
            %returns the distance matrix between 2 sets of states
            %X and Z
            distsqr = slmetric_pw(X',Z','sqdist');  %matrix of distances between the states
        end
        
        
        function [distsqr] = stateActionSqDistance(obj, X, Z)
            
            statedistsqr = obj.sqDistance(X(:,1:obj.state_dim), Z(:,1:obj.state_dim));
            distsqr = statedistsqr + squareDist(X(:,(obj.state_dim+1):end),Z(:,(obj.state_dim+1):end));
            
        end
        
        function [pd] = density(obj, state, action, successor)
            % returns the density at a given successor state conditioned on
            % a given state and action
            new_state = obj.detTransit(state,action);
            pd = mvnpdf(successor, new_state, obj.noise);
        end
        
        
        
        function [rwd] = reward(obj, state, action)
            % returns the reward at a given srtate and action
            state_cost = abs(state - 3);
   
            rwd = exp(-(state_cost)); %in (0,1)
        end
        
        function [start_state, start_state_action] = getStartState(obj)
            %returns a starting state from the start state distribution
            start_state = obj.start_state_noise*randn(1);
            start_state_action = [start_state 0];
        end
        
        function [action] = uniform_action(obj,n_actions)
            if(~exist('n_actions','var'))
                n_actions = 1;
            end
            action = unifrnd(-obj.max_action, obj.max_action,n_actions,1);
        end
        
        function [state] = uniform_state(obj)
              
            [state] = unifrnd(-4, 4);
        end
        
        function [state] = reward_state(obj)
            state = 3 + unifrnd(-0.1,0.1);
        end
        
        function [new_state] = transit(obj, state, action)
            %STOCHASTIC TRANSITION
            %add some noise
            [new_state] = obj.detTransit(state, action) + obj.noise*randn(1);
        end
        
       
        
    end

    
    
    
    methods(Access = private)
        function [new_state] = detTransit(obj, state, action)
            %DETERMINISTIC TRANSITION
            action(action > 1) = obj.max_action;
            action(action < -1) = -obj.max_action;
            new_state = state + action;
            
            if(abs(new_state)>4)
                new_state = 0;
            end
        end
        
        
    end
end