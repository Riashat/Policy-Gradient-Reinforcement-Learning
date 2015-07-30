classdef MountainCar < MDP & DistanceMeasure & DistanceGradientMeasure
    
    properties(SetAccess = private)
        velocityScalingParameter;
    end
    
    methods(Access = public)
        
        function obj = MountainCar(noise, gamma, H, actionDiscCount)
            
            %the classic mountain car domain has fixed parameters...
            actionDiscCount = 3;
            noise = 0.01;
            H=100;          
            start_state_noise = [0 0]; 
            state_dim = 2;
            action_dim = 1;
            max_action = 1;
            %for discretizing
            min_disc_act = -max_action;
            max_disc_act = max_action;
            
            max_state = [0.7 0.07 ]; % 
            min_state = [-1.2 -0.07 ]; %
            name = 'MountainCar';
            
            obj = obj@MDP(gamma, H, noise, start_state_noise, state_dim, action_dim, max_action,...
                min_disc_act, max_disc_act, max_state, min_state, actionDiscCount, name);
            obj.actionDiscCount = actionDiscCount;
            obj.discAct = 1;
            obj.all_actions = obj.discretizeActions();             
            obj.velocityScalingParameter = 0.1;
        end
        
        function [inputFeatureFactory, outputFeatureFactory] = getFeatureFactory(obj, OPTS)
   
        end
        
        function kernelFactory = getKernelFactorty(obj)
            kernelFactory = MountainCarKernelFactory(obj);
        end
        
        function [distsqr] = sqDistance(obj, X, Z)
            distsqr =  squareDist(X(:,1),Z(:,1))  +  squareDist(X(:,2),Z(:,2))/obj.velocityScalingParameter^2;
        end
        
        
        function [distsqr] = stateActionSqDistance(obj, X, Z)
            statedistsqr = obj.sqDistance(X(:,1:obj.state_dim), Z(:,1:obj.state_dim));
            distsqr = statedistsqr + squareDist(X(:,(obj.state_dim+1):end),Z(:,(obj.state_dim+1):end));
        end
        
        function [distsqr] = sqDistanceElementWise(obj, X, Z)
            distsqr =  squareDist(X(:,1),Z(:,1))  +  squareDist(X(:,2),Z(:,2))/obj.velocityScalingParameter^2;
        end
        
        function [ grads ] = sqDistanceGrad(obj, X, Z)
        end
        
        function [pd] = density(obj, state, action, successor)            
            [newpos, newvel] = obj.detTransit(state,action);
            new_state = [newpos newvel];

             pd = exp(-obj.sqDistance(successor,new_state)/(obj.noise^2));
        end
        
        function [rwd] = reward(obj, states, actions)
            positionCosts = (bsxfun(@minus ,states(:,1) , 0.6 )).^2;
            costs = positionCosts;
            rwd = exp( - costs/(2*0.25^2)  ) ;
            %rwd = exp( - costs  ) ;
        end
        
        
        function [start_state, start_state_action] = getStartState(obj)
            %returns a starting state from the start state distribution
            position = -0.5 + obj.start_state_noise(1)*randn(1);
            velocity = 0 + obj.start_state_noise(2)*randn(1);
            start_state = [position velocity];
            start_state_action = [start_state 0];
        end
        
        function [action] = uniform_action(obj, n_actions)
            if(~exist('n_actions','var'))
                n_actions = 1;
            end
            action = randsample(obj.all_actions,n_actions);
        end
        
        function [state] = uniform_state(obj, varargin)
            if isempty(varargin)
                pos = unifrnd(obj.min_state(1), obj.max_state(1));
                vel = unifrnd(obj.min_state(2),obj.max_state(2));
                state = [ pos  vel ];
            else
                minState = -varargin{1};
                maxState = varargin{1};
                pos = unifrnd(minState(1), maxState(1));
                vel = unifrnd(minState(2), maxState(2));
                state = [ pos  vel ];
            end
        end
        
        function [state] = reward_state(obj)

        end
        
        function visualiseSingleTrajectory(obj, OPTS)

        end
        
        function visualisePolicyAndExplorationData(obj, agent, OPTS)

         end
            

  
        
         function visualiseParameterisedAgent(obj, agent)

        end
        
        function visualiseTrajectories(obj, dataGenerator, figNo)

        end
        
        function [new_state] = transit(obj, state, action)
            %STOCHASTIC TRANSITION            
            [new_pos, new_vel] = obj.detTransit(state, action);            
            %add some noise
            new_pos = new_pos + obj.noise*randn(1);
            new_vel = new_vel + obj.noise*obj.velocityScalingParameter*randn(1);
            
            % Inelastic walls
            if ( new_pos < obj.min_state(1) || new_pos > obj.max_state(1) )
                new_vel = 0;
            end
            
            new_vel = min(max(new_vel, obj.min_state(2)), obj.max_state(2));
            new_pos = min(max(new_pos, obj.min_state(1)), obj.max_state(1));
            
            new_state = [new_pos new_vel];
        end
        
           end   
        

    
    methods(Access = protected)
        function [position, velocity, actions] = processTrajectoriesForVisualisation(~, trajectories)

        end
    end
    
    methods(Access = private)
        
        function [new_pos, new_vel] = detTransit(obj, state, action)

            [~ , inds] = min(abs( action - repmat(obj.all_actions', size(action,1), 1) ), [], 2);
            action = obj.all_actions(inds);
            
            pos = state(:,1);
            vel = state(:,2);
            new_pos = pos + vel;
            new_vel = vel +  action*0.001 + obj.getAccelerationDueToG(pos);

            if ( new_pos < obj.min_state(1) || new_pos > obj.max_state(1) )
                new_vel = 0;
            end
            
            new_vel = min(max(new_vel, obj.min_state(2)), obj.max_state(2));
            new_pos = min(max(new_pos, obj.min_state(1)), obj.max_state(1));            
            vel = new_vel;
            pos = new_pos;
            new_pos = pos + vel;
            new_vel = vel +  action*0.001 + obj.getAccelerationDueToG(pos);

            if ( new_pos < obj.min_state(1) || new_pos > obj.max_state(1) )
                new_vel = 0;
            end
            
            new_vel = min(max(new_vel, obj.min_state(2)), obj.max_state(2));
            new_pos = min(max(new_pos, obj.min_state(1)), obj.max_state(1));
                       
        end
        
        function acceleration = getAccelerationDueToG(~, position)
            acceleration = cos(3.*position).*(-0.0025);
        end
        
        function height = getHeight(~, position)
            height = sin(3.*position)*0.005 + 0.005;
        end
        
    end
end