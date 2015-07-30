classdef Pendulum < MDP & DistanceMeasure & DistanceGradientMeasure
    
    properties(SetAccess = private)
        dt = 0.05;    % the time step for discrete dynamics. 0.05 is certainly fine enough to learn good policies
        m = 1;       % [kg] mass of Pendulum  (was set to 1)
        mu = 0.05;   % []  friction (was set to 0.05)
        l = 1;       % [m] Pendulum length
        g = 9.81;    % [m/s^2]  acceleration of gravity
        velocityScalingParameter = 2;
        isRKSolver

    end
    
    methods(Access = public)
        
        function obj = Pendulum(noise, gamma, H, actionDiscCount)
             actionDiscCount = 9;
             H=400;
             noise =  0.02;
             gamma = 0.99;
            
            start_state_noise = [0 0]; %0.1 if you want a bit of initial noise in the system
            state_dim = 2;
            action_dim = 1;
            max_action = 3;
            %for discretizing
            min_disc_act = -max_action;
            max_disc_act = max_action;
            %for sampling
            max_state = [pi 4*pi];
            min_state = [-pi -4*pi];
            
            name = 'Pendulum';
            %spatial_dim = 1;
            
            obj = obj@MDP(gamma, H, noise, start_state_noise, state_dim, action_dim, max_action,...
                min_disc_act, max_disc_act, max_state, min_state, actionDiscCount, name);
            obj.actionDiscCount = actionDiscCount;
            obj.all_actions = obj.discretizeActions();
            obj.discAct = 0; %continuous
            
           
            
        end
        
        function [discrete_actions] = discretizeActions(obj)
            %only works for 1d actions will have to be overwritten in
            %higher dim
     %       discrete_actions = [-3 -1.5 -0.8 -0.4 -0.2 -0.1 0 0.1 0.2 0.4 0.8 1.5 3]';
            discrete_actions = [-3 -1 -0.5 -0.2  0 0.2 0.5 1 3]';
        end

        
        function [distsqr] = sqDistance(obj, X, Z)
            %returns the distance matrix between 2 sets of vectors of states
            %X and Z - FOR WHEN MEMORY IS NOT AT A PREMIUM
            absDiff = abs(bsxfun(@minus, X(:,1), Z(:,1)'));
          distsqr = (min( 2*pi-absDiff, absDiff )).^2 + (squareDist(X(:,2),Z(:,2))/obj.velocityScalingParameter^2);
            % distsqr = (min( 2*pi-absDiff, absDiff )).^2 + ((X(:,2)-Z(:,2)').^2/obj.velocityScalingParameter^2);
             
             %distsqr = (min( 2*pi-absDiff, absDiff )).^2 + (abs(bsxfun(@minus,X(:,2),Z(:,2)')).^2/obj.velocityScalingParameter^2);
        end

        
        function [distsqr] = stateActionSqDistance(obj, X, Z)
            statedistsqr = obj.sqDistance(X(:,1:obj.state_dim), Z(:,1:obj.state_dim));
            distsqr = statedistsqr + squareDist(X(:,(obj.state_dim+1):end),Z(:,(obj.state_dim+1):end));
            
        end
        

        

        
        function [pd] = density(obj, state, action, successor)
            % returns the density at a given successor state conditioned on
            % a given state and action. either state-action, or successor
            % can be a vector of several but not both
            [new_angle, new_angvel] = obj.detTransit(state,action);
            new_state = [new_angle new_angvel];
            pd = exp(-obj.sqDistance(successor,new_state)/(obj.noise^2));
        end
        
        function [ new_state rwd ] = engine(obj, state, action)
            % full transition including the reward
            new_state = obj.transit(state,action);
            rwd  = obj.reward(state, action);
        end
        
        function [rwd] = reward(obj, states, actions)
            % returns the reward at a given srtate and action
            angle_costs = 1*(states(:,1)).^2;
            %angvel_costs = 0.1*(state(:,2)).^2;  % higher cost on large angular velocity?
            %action_costs = 0.1*abs(action).^2;   %penalizes high actions
            %action_costs = 0.02*abs(actions).^3; %really penalizes high actions
            costs = angle_costs;% + angvel_costs + action_costs;
            
            %             OLD
            %             clear angvel_costs  angle_costs action_costs
            %
            %             maxcosts = 3.14^2 + 0.1*15^2 + 0.1*5^2 ; %just to try and make most rewards in [0,1]
            %             costs  =  costs ./ maxcosts;
            %             rwd = 1 - costs;
            
            bwdth = 2;
            rwd = exp(-costs/bwdth);
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
        
        
        function [state] = uniform_state(obj, varargin)
            if isempty(varargin)
                
                 maximum_state = [pi 4*pi];
                 minimum_state = [-pi -4*pi];
                angle = unifrnd(minimum_state(1), maximum_state(1));
                ang_vel = unifrnd(minimum_state(2),maximum_state(2));
                 
%                  
%                 angle = unifrnd(obj.min_state(1), obj.max_state(1));
%                 ang_vel = unifrnd(obj.min_state(2),obj.max_state(2));
                state = [ angle  ang_vel ];
            else
                minState = -varargin{1};
                maxState = varargin{1};
                angle = unifrnd(minState(1), maxState(1));
                ang_vel = unifrnd(minState(2), maxState(2));
                state = [ angle  ang_vel ];
            end
        end
        
        
        function [start_state, start_state_action] = getStartState(obj)
            %returns a starting state from the start state distribution
            angle = pi + obj.start_state_noise(1)*randn(1);
            modAngle = mod( angle + pi , 2*pi ) - pi; %put in [-pi,pi)
            vel = 0 + obj.start_state_noise(2)*randn(1);
            start_state = [modAngle vel];
            start_state_action = [start_state 0];
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
        
        
        function [xyz] = visualiseTrajectories(obj, det)
        end
        
        
    end

    methods(Access = private)
        
        function [new_angle new_angvel] = detTransit(obj, state, action)
            %DETERMINISTIC TRANSITION
            
            % creates transitions function for the **deterministic** Pendulum
            
            
            % state is an n by 2 array. n states and each state is 2d
            % (angle,angular_velocity)
            % angle should be in [-pi,pi)
            % angle 0 is the top of the Pendulum swing
            % action is n by 1 and is the torque applied to the Pendulum
            
            %clip the action at [-8,8]
            %this ensures the learner cannot just apply very high torque to
            %lift the Pendulum straight up, and must do the swing up
            %in a real system such high torque might be impossible or
            %damaging to the system
            
            if abs(action) > obj.max_action;
                action = obj.max_action*sign(action);
            end
            
            % Make two steps to half the sample rate
            %             [new_angle, new_angvel] = obj.solverOrig(state, action);
            %             [new_angle, new_angvel] = obj.solverOrig([new_angle, new_angvel], action);
            if obj.isRKSolver
                [new_angle, new_angvel] = obj.solverRK4(state, action);
                [new_angle, new_angvel] = obj.solverRK4([new_angle, new_angvel], action);
            else
                [new_angle, new_angvel] = obj.solverOrig(state, action);
                [new_angle, new_angvel] = obj.solverOrig([new_angle, new_angvel], action);
            end
            
%             [new_angle, new_angvel] = obj.solverOde45(obj, state, action);
%             [new_angle, new_angvel] = obj.solverOde45([new_angle, new_angvel], action);
        end
        
        function [new_angle, new_angvel] = solverOrig(obj, state, action)
            % Mark's
            new_angle = mod( state(:,1) + obj.dt.*state(:,2) + pi , 2*pi ) - pi; %this is the angle and needs to be in [-pi,pi)
            new_angvel = state(:,2) + obj.dt.*(-obj.mu.*state(:,2) + (obj.m*obj.g*obj.l).*sin(state(:,1))/2 + action )./(obj.m*obj.l^2/3);
            
            % Tom's
            %             new_angle = mod( state(:,1) + obj.dt.*state(:,2) + pi , 2*pi ) - pi; %this is the angle and needs to be in [-pi,pi)
            %             new_angvel = state(:,2) + obj.dt.*(-obj.mu.*state(:,2) + (obj.m*obj.g*obj.l).*sin(state(:,1)) + action )./(obj.m*obj.l^2);
            %            dy(1) = mod( GRID(s,1) + dt.*GRID(s,2) +pi , 2*pi ) - pi; %this is the angle and needs to be in [-pi,pi)
            %            dy(2) = GRID(s,2) + dt.*(-mu.*GRID(s,2) + (m*g*l).*sin(GRID(s,1)) + actions(a))./(m*l^2);
        end
        
        
        
    end
end