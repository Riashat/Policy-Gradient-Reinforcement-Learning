classdef CartPole < MDP & DistanceMeasure & DistanceGradientMeasure
    
    properties(SetAccess = private)
        dt = 0.1;    % the time step for discrete dynamics. 0.05 is certainly fine enough to learn good policies
        M = 8
        m = 2;       % [kg] mass of Pendulum  (was set to 1)
%        mu = 0.05;   % []  friction (was set to 0.05)
        l = 0.5;       % [m] Pendulum length
        g = 9.8;    % [m/s^2]  acceleration of gravity
        alpha 
        velocityScalingParameter = 2;
        actionScaling
    end
    
    methods(Access = public)
        
        function obj = CartPole()
            actionDiscCount = 3;
            H=100;  %400
            noise =  0;
            gamma = 0.99;
            
            start_state_noise = [0 0]; %0.1 if you want a bit of initial noise in the system
            state_dim = 2;
            action_dim = 1;
            max_action = 50;
            %for discretizing
            min_disc_act = -max_action;
            max_disc_act = max_action;
            %for sampling
            max_state = [pi 4*pi];
            min_state = [-pi -4*pi];
            
            name = 'CartPole';
            spatial_dim = 1;
            
            obj = obj@MDP(gamma, H, noise, start_state_noise, state_dim, action_dim, max_action,...
                min_disc_act, max_disc_act, max_state, min_state, actionDiscCount, name, spatial_dim);
            obj.actionScaling = max_action*2;
            obj.alpha = 1/(obj.m+obj.M);
        end
        
        function [discrete_actions] = discretizeActions(obj)
            discrete_actions = [-50 0 50]';
        end
        
        function kernelFactory = getKernelFactorty(obj)
            kernelFactory = PendKernelFactory(obj);
        end
        
        function stateScalingVector = getStateScaling(obj, scaleConst)
            stateScalingVector = [1 1/(obj.velocityScalingParameter^2)]/(scaleConst^2);
        end
        
        function [distsqr] = sqDistance(obj, X, Z)
            %returns the distance matrix between 2 sets of vectors of states
            %X and Z - FOR WHEN MEMORY IS NOT AT A PREMIUM
            scaling = obj.getStateScaling(1);
            absDiff = abs(bsxfun(@minus, X(:,1), Z(:,1)'));
            distsqr = scaling(1)*(min( 2*pi-absDiff, absDiff )).^2 + scaling(2)*squareDist(X(:,2),Z(:,2));
        end
        
        function [distsqr] = stateActionSqDistance(obj, X, Z)
            statedistsqr = obj.sqDistance(X(:,1:obj.state_dim), Z(:,1:obj.state_dim));
            distsqr = statedistsqr + squareDist(X(:,(obj.state_dim+1):end),Z(:,(obj.state_dim+1):end));
        end
        
        function [distsqr] = sqDistanceElementWise(obj, X, Z)
            %returns the distance matrix between 2 sets of vectors of states
            %X and Z - FOR WHEN MEMORY IS TIGHT
            scaling = obj.getStateScaling(1);
            distsqr = bsxfun( @plus, scaling(1)*elementWiseMinThenSqrd( abs(bsxfun(@minus, X(:,1), Z(:,1)')), 2*pi ),...
                scaling(2)*squareDist(X(:,2),Z(:,2)) );
        end
        
        function [ grads ] = sqDistanceGrad(~, X, Z)
            %returns the gradient w.r.t. X of the (squared) distance function
            %X can be a matrix of several inout vectors
            for j =1:length(Z(:,1))
                deltaAngle = (X(:,1) - repmat( Z(j,1), length(X(:,1)),1 ));
                angGrad = 2*(min(2*pi - abs(deltaAngle),abs(deltaAngle)).*sign(pi - abs(deltaAngle)).*sign(deltaAngle));
                velGrad = 2*(1/4)*((X(:,2) - repmat(Z(j,2), length(X(:,2)), 1)));
                grad = [angGrad velGrad];
                grads(j,:,:) = grad;
            end
            
        end
        
        function [pd] = density(obj, state, action, successor)
            % returns the density at a given successor state conditioned on
            % a given state and action. either state-action, or successor
            % can be a vector of several but not both
            [new_angle, new_angvel] = obj.detTransit(state,action);
            new_state = [new_angle new_angvel];
            pd = exp(-obj.sqDistance(successor,new_state)/(obj.noise^2));
        end
        
        function [ new_state, rwd ] = engine(obj, state, action)
            % full transition including the reward
            new_state = obj.transit(state,action);
            rwd  = obj.reward(state, action);
        end
        
        function [rwd] = reward(obj, states, actions)
            % returns the reward at a given srtate and action
            rwd = 0.5*(1+ cos(states(:,1)));
        end
        
        function [trajectories] = pull_trajs(obj, hpolicy, trajCount, trajLength)
            %given a policy (a handle to a function producing actions when given a state)
            %this function returns 'n_traj' trajectories of length 'length'
            
            states = NaN(trajCount, trajLength, obj.state_dim);
            actions = NaN(trajCount, trajLength, obj.action_dim);
            rwds = NaN(trajCount, trajLength);
            
            for i=1:trajCount
                
                if rem(i,10000) == 0
                    fprintf(['\Policy Trajectory Count = ', num2str(i),'\n']);
                end
                
                states(i,1,:) = obj.getStartState();
                
                for j=2:trajLength
                    actions(i,j-1,:) = hpolicy(squeeze(states(i,j-1,:))');
                    [states(i,j,:), rwds(i,j-1)] = obj.engine(states(i,j-1,:),actions(i,j-1,:));
                end
                actions(i,trajLength,:) = hpolicy(squeeze(states(i,trajLength,:))');
                [~, rwds(i,trajLength)] = obj.engine(states(i,trajLength,:),actions(i,trajLength,:));
            end
            
            trajectories.states = states;
            trajectories.actions = actions;
            trajectories.rwds = rwds;
        end
        
        function [trajectories] = pull_uniform_trajs(obj, trajCount, trajLength, varargin)
            %pulls trajectories by sampling states and actions uniformly at random
            
            states = NaN(trajCount, trajLength, obj.state_dim);
            actions = NaN(trajCount, trajLength, obj.getActionDim());
            rwds = NaN(trajCount, trajLength);
            
            for i=1:trajCount
                if rem(i,10000) == 0
                    fprintf(['\nUniform Trajectory Count = ', num2str(i),'\n']);
                end
                
                states(i,1,:) = obj.uniform_state(varargin{:});
                for j=2:trajLength
                    action = obj.uniform_action();
                    if ~isempty(varargin)
                        action = action/10;
                    end
                    actions(i,j-1,:) = action;
                    [states(i,j,:), rwds(i,j-1)] = obj.engine(states(i,j-1,:),actions(i,j-1,:));
                end
                
                action = obj.uniform_action();
                if ~isempty(varargin)
                    action = action/10;
                end
                
                actions(i,trajLength,:) = action;
                [~, rwds(i,trajLength)] = obj.engine(states(i,trajLength,:),actions(i,trajLength,:));
            end
            
            trajectories.states = states;
            trajectories.actions = actions;
            trajectories.rwds = rwds;
        end
        
        function [trajectories] = pull_trajs_normStart_uniformAction(obj, trajCount, trajLength)
            %pulls trajectories by sampling states and actions uniformly at random
            states = NaN(trajCount, trajLength, obj.state_dim);
            actions = NaN(trajCount, trajLength, obj.getActionDim());
            rwds = NaN(trajCount, trajLength);
            
            for i=1:trajCount
                states(i,1,:) = obj.getStartState();
                
                for j=2:trajLength
                    actions(i,j-1,:) = obj.uniform_action();
                    [states(i,j,:), rwds(i,j-1)] = obj.engine(states(i,j-1,:),actions(i,j-1,:));
                end
                actions(i,trajLength,:) = obj.uniform_action();
                [~, rwds(i,trajLength)] = obj.engine(states(i,trajLength,:),actions(i,trajLength,:));
            end
            
            trajectories.states = states;
            trajectories.actions = actions;
            trajectories.rwds = rwds;
        end
        
        function [trajectories] = pull_trajs_uniform_start(obj, hpolicy, trajCount, trajLength)
            %given a policy (a handle to a function producing actions when given a state)
            %this function returns 'n_traj' trajectories of length 'length'
            
            states = NaN(trajCount, trajLength, obj.state_dim);
            actions = NaN(trajCount, trajLength, obj.action_dim);
            rwds = NaN(trajCount, trajLength);
            
            for i=1:trajCount
                states(i,1,:) = obj.uniform_state();
                
                for j=2:trajLength
                    actions(i,j-1,:) = hpolicy(squeeze(states(i,j-1,:))');
                    [states(i,j,:), rwds(i,j-1)] = obj.engine(states(i,j-1,:),actions(i,j-1,:));
                end
                actions(i,trajLength,:) = hpolicy(squeeze(states(i,trajLength,:))');
                [~, rwds(i,trajLength)] = obj.engine(states(i,trajLength,:),actions(i,trajLength,:));
            end
            
            trajectories.states = states;
            trajectories.actions = actions;
            trajectories.rwds = rwds;
        end
        
        function [start_state, start_state_action] = getStartState(obj)
            %returns a starting state from the start state distribution
            angle = pi + obj.start_state_noise(1)*randn(1);
            modAngle = mod( angle + pi , 2*pi ) - pi; %put in [-pi,pi)
            vel = 0 + obj.start_state_noise(2)*randn(1);
            start_state = [modAngle vel];
            start_state_action = [start_state 0];
        end
        
        function [action] = uniform_action(obj,n_actions)
            if(~exist('n_actions','var'))
                n_actions = 1;
            end
            action = unifrnd(-2*obj.getMaxAction(), 2*obj.getMaxAction(),n_actions,1);
        end
        
        function [state] = uniform_state(obj, varargin)
            if isempty(varargin)
                angle = unifrnd(obj.min_state(1), obj.max_state(1));
                ang_vel = unifrnd(obj.min_state(2),obj.max_state(2));
                state = [ angle  ang_vel ];
            else
                minState = -varargin{1};
                maxState = varargin{1};
                angle = unifrnd(minState(1), maxState(1));
                ang_vel = unifrnd(minState(2), maxState(2));
                state = [ angle  ang_vel ];
            end
        end
        
        function [state] = reward_state(obj)
            
            angle = unifrnd(-0.1,0.1);
            ang_vel =unifrnd(-0.3,0.3);
            state = [ angle  ang_vel ];
            
        end
        
        function [new_state] = transit(obj, state, action)
            %STOCHASTIC TRANSITION
            
            [new_angle, new_angvel] = obj.detTransit(state, action);
            %add some noise
            new_angle = new_angle + obj.noise*randn(1);
            new_angvel = new_angvel + obj.noise*randn(1)*obj.velocityScalingParameter;
            
            new_angle = mod( new_angle + pi , 2*pi ) - pi; %put in [-pi,pi)
            new_state = [new_angle new_angvel];
        end
        
        function visualiseTrajectories(obj, dataGenerator, figNo)
            [x, y, thetas, omegas, actions] = obj.processTrajectoriesForVisualisation(dataGenerator.pullTrajectories(obj.H-1,1));
            runPendulumMovie( x, y, thetas, omegas, actions, figNo )
        end
        
        function visualiseSingleTrajectory(obj, OPTS)
            rng('shuffle');
            % Run single trajectory Animation
            agentStateKer = obj.getKernelFactorty().makeNewStateKernel(OPTS.agent.stateBandwidth);
            controller = ParametricController(OPTS.agent.isDeterministic, OPTS.agent.n_centres, OPTS.agent.initSigma, agentStateKer, obj, OPTS.agent.policyFilePath);
            [ ~, trajectoryGenerator, ~, ~ ] = buildDataGeneratorOfType( 'PolicySampler', controller, obj, 1, OPTS.mdp.H, OPTS);
            obj.visualiseTrajectories(trajectoryGenerator, 556);
            disp(trajectoryGenerator.estimatedRewardFor(@controller.policy));
            
            % Weight/action surface
            obj.visualiseParameterisedAgent(controller);
            
            %(s,a)
            obj.visualisePolicyAndExplorationData(controller, OPTS);
        end
        
        function visualisePolicyAndExplorationData(obj, agent, OPTS)
            
            [ ~, ~, policyDataSampler, explorationDataSampler ] = buildDataGeneratorOfType( OPTS.data.sampler.type, agent, obj, OPTS.data.M, OPTS.data.T, OPTS  );
            
            
            [Z_pol,S_pol] = policyDataSampler.pullData(400,1);
            Z_exp = [];
            S_exp = [];
            if ~isempty(explorationDataSampler)
                [Z_exp, S_exp] = explorationDataSampler.pullData(400,1);
            end
            size(Z_pol)
            size(Z_exp)
            
            figure(777)
            plot3(Z_pol(:,1),Z_pol(:,2),Z_pol(:,3), 'b.');
            hold on
            plot3(S_pol(:,1),S_pol(:,2),Z_pol(:,3), 'b.');
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
        
    end
    
    methods(Access = protected)
        function [x, y, thetas, omegas, actions] = processTrajectoriesForVisualisation(~, trajectories)
            [M, T] = size(trajectories.states(:,:,1));
            thetas = reshape(trajectories.states(:,:,1)', [1 M*T]);
            omegas = reshape(trajectories.states(:,:,2)', [1 M*T]);
            actions = reshape(trajectories.actions', [1 M*T]);
            x = sin(thetas);
            y = cos(thetas);
        end
    end
    
    methods(Access = private)
        
        function [new_angle, new_angvel] = detTransit(obj, state, action)
            if abs(action) > obj.max_action;
                action = obj.max_action*sign(action);
            end
            actNoise = unifrnd(-10*ones(size(action,1),1),10*ones(size(action,1),1));
            action = action + actNoise;
                [new_angle, new_angvel] = obj.solverOrig(state, action);
        end
        
        function [new_angle, new_angvel] = solverOrig(obj, state, action)

            new_angle = mod( state(:,1) + obj.dt.*state(:,2) + pi , 2*pi ) - pi; %this is the angle and needs to be in [-pi,pi)
            acceleration = (obj.g*sin(state(:,1)) - obj.alpha*obj.m*obj.l*(state(:,2).^2).*sin(state(:,1)*2) -obj.alpha*cos(state(:,1)).* action )./(4*obj.l / 3 - obj.alpha*obj.m*obj.l*(cos(state(:,1))).^2);
            new_angvel = state(:,2) + obj.dt.*acceleration;
        
        end
        
    end
end