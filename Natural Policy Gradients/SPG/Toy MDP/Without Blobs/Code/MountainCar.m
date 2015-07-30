classdef MountainCar < MDP & DistanceMeasure & DistanceGradientMeasure
    
    properties(SetAccess = private)
        velocityScalingParameter;
    end
    
    methods(Access = public)
        
        function obj = MountainCar(noise, gamma, H, actionDiscCount)
            
            %the classic mountain car domain has fixed parameters...
            actionDiscCount = 3;
            noise = 0.02;
            H=100;
            
            start_state_noise = [0 0]; %0.1 if you want a bit of initial noise in the system
            state_dim = 2;
            action_dim = 1;
            max_action = 1;
            %for discretizing
            min_disc_act = -max_action;
            max_disc_act = max_action;
            
            
            %for sampling
            %state is [position velocity]
            max_state = [0.7 0.07 ]; % should be max_state = [0.6 0.07 ]; but put the wall a little beyond the goal state
            min_state = [-1.2 -0.07 ]; %
            name = 'MountainCar';
            
            obj = obj@MDP(gamma, H, noise, start_state_noise, state_dim, action_dim, max_action,...
                min_disc_act, max_disc_act, max_state, min_state, actionDiscCount, name);
            obj.actionDiscCount = actionDiscCount;
            obj.discAct = 1;
            obj.all_actions = obj.discretizeActions(); %will be [-1,0,1] in the standard set up
            
            obj.velocityScalingParameter = 0.1;
        end
        
        function [inputFeatureFactory, outputFeatureFactory] = getFeatureFactory(obj, OPTS)
            inputMesh = OPTS.data.inputMeshPointsCount;
            outputMesh = OPTS.data.outputMeshPointsCount;
            minState = [obj.min_state(1) obj.min_state(2)];
            maxState = [obj.max_state(1) obj.max_state(2)];
            inputFeatureFactory = GaussFeatureMapFactory([minState 3*obj.min_disc_act], [maxState 3*obj.max_disc_act], inputMesh, OPTS.data.inputFeatureSigma, @obj.stateActionSqDistance, OPTS);
            outputFeatureFactory = GaussFeatureMapFactory(minState, maxState, outputMesh, OPTS.data.outputFeatureSigma, @obj.sqDistance, OPTS);
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
            disp('MOUNTAIN CAR sqDistanceGrad(): NOT IMPLEMENTED YET');
            %returns the gradient w.r.t. X of the (squared) distance function
            %X can be a matrix of several inout vectors
            
            %NEEDS TO BE WRITTEN FOR GRADIENT METHODS
        end
        
        function [pd] = density(obj, state, action, successor)
            % returns the density at a given successor state conditioned on
            % a given state and action
            
            [newpos, newvel] = obj.detTransit(state,action);
            new_state = [newpos newvel];
            %pd = exp(-obj.(successor,new_state)/(obj.noise^2));
            %            pd = mvnpdf(successor,new_state,[obj.noise , obj.velocityScalingParameter*obj.noise]);
             pd = exp(-obj.sqDistance(successor,new_state)/(obj.noise^2));
        end
        
        function [rwd] = reward(obj, states, actions)
            % returns the reward at a given srtate and action
            %rwd = (states(:,1)>=0.55).*exp( - ((bsxfun(@minus ,states(:,1) , 0.6 ).*20).^2)  ) ;
            %rwd = exp( - ((bsxfun(@minus ,states(:,1) , 0.6 )).^2)/(2*0.25^2)  ) ;
            %actionCosts = (abs(actions)>obj.max_action).*0.001.*abs(actions).^2;
            positionCosts = (bsxfun(@minus ,states(:,1) , 0.6 )).^2;
            %costs = actionCosts + positionCosts;
            costs = positionCosts;
            rwd = exp( - costs/(2*0.25^2)  ) ;
        end
        
        
        function [start_state, start_state_action] = getStartState(obj)
            %returns a starting state from the start state distribution
            pos = -0.5 + obj.start_state_noise(1)*randn(1);
            vel = 0 + obj.start_state_noise(2)*randn(1);
            start_state = [pos vel];
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
            vel = unifrnd(obj.min_state(2),obj.max_state(2));
            state = [0.6 vel];
        end
        
        function visualiseSingleTrajectory(obj, OPTS)
            rng('shuffle');
            % Run single trajectory Animation
            agentStateKer = obj.getKernelFactorty().makeNewStateKernel(OPTS.agent.stateBandwidth);
            controller = ParametricController(OPTS.agent.isDeterministic, OPTS.agent.n_centres, OPTS.agent.initSigma, agentStateKer, obj, OPTS.agent.policyFilePath);
            [ ~, trajectoryGenerator, ~, ~ ] = buildDataGeneratorOfType( 'PolicySampler', controller, obj, [], [], OPTS);

            obj.visualiseTrajectories(trajectoryGenerator, 556);
            disp(trajectoryGenerator.estimatedRewardFor(@controller.policy));
            
            % Weight/action surface
            obj.visualiseParameterisedAgent(controller);
            
            %(s,a)
            obj.visualisePolicyAndExplorationData(controller, OPTS);
        end
        
        function visualisePolicyAndExplorationData(obj, agent, OPTS)
            
            [ ~, ~, policyDataSampler, explorationDataSampler ] = buildDataGeneratorOfType( OPTS.data.sampler.type, agent, obj, [], [], OPTS  );
            [Z_pol,S_pol] = policyDataSampler.pullData(300, 2);

            Z_exp = [];
            S_exp = [];
            if ~isempty(explorationDataSampler)
                [Z_exp, S_exp] = explorationDataSampler.pullData(300, 2);
            end
            
            figure(777)
            plot3(Z_pol(:,1), Z_pol(:,2), Z_pol(:,3), 'b.');
            hold on
            plot3(S_pol(:,1), S_pol(:,2), Z_pol(:,3), 'b.');
            hold on
            if ~isempty(explorationDataSampler)
                plot3(Z_exp(:,1),Z_exp(:,2),Z_exp(:,3), 'r.');
                hold on
                plot3(S_exp(:,1),S_exp(:,2),Z_exp(:,3), 'r.');
            end
            hold off
            xlabel('\theta');
            ylabel('\omega');
            zlabel('action')
            
        end
        
        function visualiseParameterisedAgent(obj, agent)
            centres = agent.centres;
            weights = agent.weights;
            
            figNo = 101010;
            figure(figNo)
            plot3(centres(:,1), centres(:,2), weights, '.g')
            axis([obj.min_state(1) obj.max_state(1)...
                obj.min_state(2) obj.max_state(2)...
                min(weights), max(weights)]);
            
            xlabel('position');
            ylabel('velocity');
            zlabel('weight')
            title('Weight Surface')
            
            figure(figNo + 1)
            [pos, vel] = meshgrid(linspace(obj.min_state(1),obj.max_state(1),100), linspace(obj.min_state(2),obj.max_state(2),100));
            actions = agent.policy([pos(:) vel(:)]);
            plot3(pos(:), vel(:), actions, '.g')
            axis([obj.min_state(1) obj.max_state(1)...
                obj.min_state(2) obj.max_state(2)...
                min(actions), max(actions)]);
            
            xlabel('position');
            ylabel('velocity');
            zlabel('action')
            title('Action Surface')
        end
        
        function visualiseTrajectories(obj, dataGenerator, figNo)
%             figure(figNo + 1)
%             title('Acceleration due to g')
%             plot(linspace(obj.min_state(1),obj.max_state(1),100), obj.getAccelerationDueToG(linspace(obj.min_state(1),obj.max_state(1),100)), 'r',...
%                 linspace(obj.min_state(1),obj.max_state(1),100), obj.getHeight(linspace(obj.min_state(1),obj.max_state(1),100)), 'k');
%             legend({'Acceleration' 'Hill'})
            [position, velocity, actions] = obj.processTrajectoriesForVisualisation(dataGenerator.pullTrajectories(obj.H-1, 1));
            
            hillX = linspace(obj.min_state(1),obj.max_state(1),100);
            hillY = obj.getHeight(hillX);
            runMountainCarMovie([hillX' hillY'], [position' obj.getHeight(position)'], velocity, actions, figNo );
        end
        
        function [new_state] = transit(obj, state, action)
            %STOCHASTIC TRANSITION
            
            [new_pos, new_vel] = obj.detTransit(state, action);
            
            %posDiff = abs(new_pos - state(1))
            %velDiff = abs(new_vel - state(2))
            
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
            [M, T] = size(trajectories.states(:,:,1));
            position = reshape(trajectories.states(:,:,1)', [1 M*T]);
            velocity = reshape(trajectories.states(:,:,2)', [1 M*T]);
            actions = reshape(trajectories.actions', [1 M*T]);
        end
    end
    
    methods(Access = private)
        
        function [new_pos, new_vel] = detTransit(obj, state, action)
            %DETERMINISTIC TRANSITION
            [~ , inds] = min(abs( action - repmat(obj.all_actions', size(action,1), 1) ), [], 2);
            action = obj.all_actions(inds);
            
            pos = state(:,1);
            vel = state(:,2);
            new_pos = pos + vel;
            new_vel = vel +  action*0.001 + obj.getAccelerationDueToG(pos);
            
            
            % Inelastic walls
            if ( new_pos < obj.min_state(1) || new_pos > obj.max_state(1) )
                new_vel = 0;
            end
            
            new_vel = min(max(new_vel, obj.min_state(2)), obj.max_state(2));
            new_pos = min(max(new_pos, obj.min_state(1)), obj.max_state(1));
            
            % do one more step forwards
            
            vel = new_vel;
            pos = new_pos;
            new_pos = pos + vel;
            new_vel = vel +  action*0.001 + obj.getAccelerationDueToG(pos);
            
            % Inelastic walls
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
            % Scaled so we can visualise
            height = sin(3.*position)*0.005 + 0.005;
        end
        
    end
end