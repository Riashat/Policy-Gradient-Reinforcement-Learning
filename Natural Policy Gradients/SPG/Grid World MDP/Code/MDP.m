classdef MDP < handle
    
    properties(Access = public, Constant)
        REWARD_PENALTY = -1e6;
    end
    
    properties(SetAccess = protected, GetAccess = public)
        gamma;
        H;
        noise;
        start_state_noise;
        state_dim;
        action_dim;
        max_action;
        %for discretizing
        min_disc_act;
        max_disc_act;
        actionDiscCount;
        %for sampling
        max_state;
        min_state;
        discount_vec;
        name;
        all_actions = NaN; %a discretized set of actions
        discAct; %are actions discrete or continuous?
        isDebug = false;
        spatial_dim;
    end
    
    methods(Access = public)
        
%         function obj = MDP(noise, start_state_noise, state_dim, action_dim, max_action,...
%                 actionDiscCount, name)
%             
            %for pendulum    
%              function obj = MDP(gamma, H, noise, start_state_noise, state_dim, action_dim, max_action,...
%                  min_disc_act, max_disc_act, max_state, min_state, actionDiscCount, name)
%             
            
%             %for ConntGirdWorld
%             function obj = MDP(noise,H,gamma,  start_state_noise, state_dim, action_dim, min_action, max_action,...
%                actionDiscCount, name)
           
           function obj = MDP(gamma, H, noise, start_state_noise, state_dim, action_dim, max_action,...
                min_disc_act, max_disc_act, max_state, min_state, actionDiscCount, name, spatial_dim);
           
            
            
            %typical MDP params
            obj.gamma = gamma;
            obj.H = H;
            obj.noise = noise;
            obj.start_state_noise = start_state_noise;
            obj.state_dim = state_dim;
            obj.action_dim = action_dim;
            obj.max_action = max_action;
            %for discretizing
            obj.min_disc_act = min_disc_act;
            obj.max_disc_act = max_disc_act;
            obj.actionDiscCount = actionDiscCount;
            %for sampling
            obj.max_state = max_state;
            obj.min_state = min_state;
            obj.name = name;
            %obj.spatial_dim = spatial_dim;
        end
        
        function [dist] = distForNN(obj,X,Z)
            %this is just a distance function for nnsearch algorithm which
            %requires that things are the transpose of the sqDistance fn and
            %also sqrted
            dist = sqrt(obj.sqDistance(X,Z)');
        end
        
        function [inputFeatureFactory, outputFeatureFactory] = getFeatureFactory(obj, OPTS)
            statesFeatureFactory = GaussStatesFeatureMapFactory(obj, OPTS.data.features.inputStateDim, OPTS.data.features.input.sigma, OPTS);
            actionsFeatFactory = GaussActionsFeatureMapFactory(obj, OPTS.data.features.inputActionDim, OPTS.data.features.input.sigma, OPTS);
            inputFeatureFactory = GaussStateActionsFeatureMapFactory(obj, statesFeatureFactory, actionsFeatFactory);
            outputFeatureFactory = GaussStatesFeatureMapFactory(obj, OPTS.data.features.outputStateDim, OPTS.data.features.output.sigma, OPTS);
        end
        
        function centres = getDeterministicCentres(obj, nCentres)
            n = ceil(sqrt(nCentres));
            centres = zeros(n^2, obj.state_dim);
            
            [Dists, Vels] = meshgrid(linspace(obj.min_state(1), obj.max_state(1), n), linspace(obj.min_state(2), obj.max_state(2), n));
            for i = 1:n
                for j = 1:n
                    centres(sub2ind([n n],i,j),:) = [Dists(i,j) Vels(i,j)];
                end
            end
            
        end
        
        function centres = getRandomCentres(obj, n_centres)
            centres =  zeros(n_centres, obj.state_dim);
            for i=1:n_centres
                centres(i,:) = obj.mdp.uniform_state();
            end
        end
        
        function [ new_state, rwd ] = engine(obj, state, action)
            % full transition including the reward
            [new_state] = obj.transit(state,action);
            rwd  = obj.reward(state, action);
        end
        
        function [state] = represent(obj,state)%return the view given a state index
            %overwritten in subclass if there is a different representation
            %(e.g. pomdps)
        end
        
        
        function [states] = uniform_states(obj, n, leaveouts)
            %pulls many states - should be overwritten in the subclass if
            %there is a better way of doing it e.g. for discrete mdps
            states = zeros(n,obj.state_dim);
            for(i=1:n)
                states(i,:) = obj.uniform_state();
            end
        end
        
        function [actions] = uniform_actions(obj, n)
            %pulls many states - should be overwritten in the subclass if
            %there is a better way of doing it e.g. for discrete mdps
            actions = zeros(n,obj.action_dim);
            for(i=1:n)
                actions(i,:) = obj.uniform_action();
            end
        end
        
        %% Trajectory generation
        function [trajectories] = pull_trajs(obj, hpolicy, trajCount, trajLength)
            %given a policy (a handle to a function producing actions when given a state)
            %this function returns 'n_traj' trajectories of length 'length'
            
            states = NaN(trajCount, trajLength, obj.state_dim);
            actions = NaN(trajCount, trajLength, obj.action_dim);
            rwds = NaN(trajCount, trajLength);
            
            for i=1:trajCount
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
        
        function [trajectories] = pull_trajs_withStartStates(obj, hPolicy, startStates, trajLength)
            trajCount = size(startStates, 1);
            states = NaN(trajCount, trajLength, obj.state_dim);
            actions = NaN(trajCount, trajLength, obj.action_dim);
            rwds = NaN(trajCount, trajLength);
            
            for i=1:trajCount
                states(i,1,:) = startStates(i,:);
                
                for j=2:trajLength
                    actions(i,j-1,:) = hPolicy(squeeze(states(i,j-1,:))');
                    [states(i,j,:), rwds(i,j-1)] = obj.engine(states(i,j-1,:),actions(i,j-1,:));
                end
                actions(i,trajLength,:) = hPolicy(squeeze(states(i,trajLength,:))');
                [~, rwds(i,trajLength)] = obj.engine(states(i,trajLength,:),actions(i,trajLength,:));
            end
            
            trajectories.states = states;
            trajectories.actions = actions;
            trajectories.rwds = rwds;
            
        end
        
        
        function [trajectories] = pull_uniform_trajs(obj, trajCount, trajLength, varargin)
            %pulls trajectories by sampling states and actions uniformly at random
            
            states = NaN(trajCount, trajLength, obj.state_dim);
            actions = NaN(trajCount, trajLength, obj.action_dim);
            rwds = NaN(trajCount, trajLength);
            
            for i=1:trajCount
                states(i,1,:) = obj.uniform_state(varargin{:});
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
        
        function [trajectories] = pull_trajs_normStart_uniformAction(obj, trajCount, trajLength)
            %pulls trajectories by sampling states and actions uniformly at random
            
            states = NaN(trajCount, trajLength, obj.state_dim);
            actions = NaN(trajCount, trajLength, obj.action_dim);
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
        
        function states = getStatesGrid(obj, gridSpacings)
            
            if length(gridSpacings) == 1
                gridSpacings = repmat(gridSpacings, [1 obj.state_dim]);
            end
            
            stateRanges = {obj.state_dim};
            for n = 1:obj.state_dim
                stateRanges{n} = linspace(obj.min_state(n), obj.max_state(n), gridSpacings(n));
            end
            states = flattenGridnd(gridnd(stateRanges));
            
        end
        
        function actions = getActionsGrid(obj, gridSpacings)
            if length(gridSpacings) == 1
                gridSpacings = repmat(gridSpacings, [1 obj.action_dim]);
            end
            
            actionRanges = {obj.action_dim};
            for n = 1:obj.action_dim
                actionRanges{n} = linspace(obj.min_disc_act(n), obj.max_disc_act(n), gridSpacings(n));
            end
            actions = flattenGridnd(gridnd(actionRanges));
        end
        
        function actions = getDiscreteActionsGrid(obj, ~)
            actionRanges = {obj.action_dim};
            for n = 1:obj.action_dim
                actionRanges{n} = obj.all_actions(:,n)';
            end
            actions = flattenGridnd(gridnd(actionRanges));
        end
        
        function zs = getZGrid(obj, gridSpacings)
            DIMS = obj.state_dim + obj.action_dim;
            
            if length(gridSpacings) == 1
                gridSpacings = repmat(gridSpacings, [1 DIMS]);
            end
            
            zRanges = {DIMS};
            for n = 1:obj.state_dim
                zRanges{n} = linspace(obj.min_state(n), obj.max_state(n), gridSpacings(n));
            end
            
            for n = obj.state_dim+1:obj.state_dim+obj.action_dim
                zRanges{n} = linspace(obj.min_disc_act(n-obj.state_dim), obj.max_disc_act(n-obj.state_dim), gridSpacings(n));
            end
            
            zs = flattenGridnd(gridnd(zRanges));
        end
        
        function zs = getZWithDiscreteActionsGrid(obj, gridSpacings)
            DIMS = obj.state_dim + obj.action_dim;
            
            if length(gridSpacings) == 1
                gridSpacings = repmat(gridSpacings, [1 DIMS]);
            end
            
            zRanges = {DIMS};
            for n = 1:obj.state_dim
                zRanges{n} = linspace(obj.min_state(n), obj.max_state(n), gridSpacings(n));
            end
            
            for n = obj.state_dim+1:obj.state_dim+obj.action_dim
                zRanges{n} = obj.all_actions(:,n-obj.state_dim)';
            end
            
            zs = flattenGridnd(gridnd(zRanges));
        end
        
%         function state = uniform_state(obj, state)
%         end
%         
        function ps = getSpatialGrid(obj, gridSpacings)
            DIMS = obj.spatial_dim;
            spatialRanges = {DIMS};
            
            if length(gridSpacings) == 1
                gridSpacings = repmat(gridSpacings, [1 DIMS]);
            end
            
            for n = 1:DIMS
                spatialRanges{n} = linspace(obj.min_state(n), obj.max_state(n), gridSpacings(n));
            end
            ps = flattenGridnd(gridnd(spatialRanges));
        end
        
    end
    
    methods(Access = public, Abstract)
    %    getKernelFactorty(obj);
        density(obj, state, action, successor);
        reward(obj, state, action);
        getStartState(obj);
    %    uniform_action(obj);
      %  uniform_state(obj);
   %     visualiseTrajectories(obj, dataGenerator);
        sqDistance(obj, X, Z);
        stateActionSqDistance(obj, X, Z);
    end
    
    methods(Access = protected, Abstract)
     %   processTrajectoriesForVisualisation(~, trajectories);
    end
    
    methods(Access = public)
        
        function [discrete_actions] = discretizeActions(obj)
            %only works for 1d actions will have to be overwritten in
            %higher dim
            discrete_actions = linspace(obj.min_disc_act, obj.max_disc_act, obj.actionDiscCount)';
        end
        
        
        function rwd = estCumRwd(obj, hpolicy_fn, n_trajectories)
            %given a policy (a handle to a function producing actions when given a state)
            %this function returns the estimated cumulative reward
            
            [trajectories] = obj.pull_trajs(hpolicy_fn, n_trajectories, obj.H);
            rwd = obj.estDiscountedReward(trajectories.rwds);
        end
        
        function discountedReward = estDiscountedReward(obj, rwds)
            if( ~exist('obj.discount_vec','var') )
                for i = 1:obj.H
                    obj.discount_vec(i) = (obj.gamma)^(i-1);
                end
                obj.discount_vec = reshape(obj.discount_vec,obj.H,1);
            end
            discountedReward = mean(rwds*obj.discount_vec);
            
        end
        
    end
    
end

