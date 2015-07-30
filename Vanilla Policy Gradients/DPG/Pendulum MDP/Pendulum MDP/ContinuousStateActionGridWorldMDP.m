    classdef ContinuousStateActionGridWorldMDP < MDP 

        %in continuous MDP - infinite number of states and actions

         properties(SetAccess = private)

         end

         methods(Access = public)

             function obj = ContinuousStateActionGridWorldMDP()
                actionDiscCount = 9;
                H=400;
                noise =  0.02;
                gamma = 0.99;           
                %********************
                start_state_noise = [0 0]; %0.1 if you want a bit of initial noise in the system
                state_dim = 2;
                %action_dim = 1;
                action_dim = 2;

                max_action = 2;
                %for discretizing
                min_disc_act = -max_action;
                max_disc_act = max_action;
                %for sampling
                max_state = [100 100];
                min_state = [0 0];
                name = 'ContinuousStateActionGridWorldMDP';
                spatial_dim = 1;

                %***************************            
                obj = obj@MDP(gamma, H, noise, start_state_noise, state_dim, action_dim, max_action,...
                    min_disc_act, max_disc_act, max_state, min_state, actionDiscCount, name);


                obj.actionDiscCount = actionDiscCount;
                %obj.all_actions = obj.discretizeActions();
                obj.discAct = 0; %continuous        
             end

             function [new_state, rwd] = engine(obj,state, action)
                [new_state] = obj.transit(state, action);
                [rwd] = obj.reward(state,action);
             end

             function [rwd] = reward(obj,state,actions)
                  bwdth=2;
                  goal_state = obj.goal_state();
                  rwd = exp(-sum((goal_state(1,:) - state(1,:)).^2));

             %define the reward with a multivariate Gaussian distribution?
             x = state(:,1);
             y = state(:,2);
    %          if (x>10 && x<11 && y > 10 && y < 11) || (x>4 && x<5 && y > 4 && y < 5) || (x>20 && x<21 && y > 20 && y < 21)
    %              rwd = 0.7;


    %          if (x>10 && x<11 && y > 10 && y < 11)
    %              rwd = 9*mvnpdf(state, [10.5 10.5], [2 2]);
    %          else
    %              if (x>4 && x<5 && y > 4 && y < 5)
    %                rwd = 9*mvnpdf(state, [4.5 5.5], [2 2]);  
    %              else
    %                  if (x>20 && x<21 && y > 20 && y < 21)
    %                      rwd = 9*mvnpdf(state, [20.5 20.5], [2 2]); 
    %              
    %          else

             %matlab multivariate normal distribution with covariance 2?

             %mean at [15 15] ? - mean around which there exists a reward blob?

             %blob at the actual goal state
             %rwd = 10*mvnpdf(state, [15 15], [2 2]);

             %end

             end


             function [goal_state] = goal_state(obj)       
              %goal_state = round(randn(1,2));
              goal_state = [10 10];
             end

             function [start_state] = getStartState(obj)
                 start_state = [0 0] + rand(1,2);
             end
             
            function [state] = uniform_state(obj)
                state1 = unifrnd(obj.min_state(1), obj.max_state(1));
                state2 = unifrnd(obj.min_state(2), obj.max_state(2));
                state = [state1 state2];
            end

             function [new_state] = transit(obj,state, action)

                [transit_state] = obj.detTransit(state, action);

                new_x = transit_state(:,1);
                new_y = transit_state(:,2);

                xnew = new_x + obj.noise*randn(size(new_x,1),1);
                ynew = new_y + obj.noise* randn(size(new_y,1),1);


                [new_state] = [xnew, ynew];


             end

             function [pd] = density(obj, state, action, successor)
                new_state = obj.detTransit(state,action);

                pd = exp(-obj.sqDistance(successor,new_state)/(obj.noise^2));
             end

            function [distsqr] = sqDistance(obj, X,Z)
                distsqr = squareDist(X(:,1),Z(:,1)) + squareDist(X(:,2), Z(:,2));
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


      %****************

            function [distsqr] = stateActionSqDistance(obj, X, Z)         
            end
    %           function [distsqr] = sqDistance(obj, X, Z)
    %             
    %         end

    function [xyz] = visualiseTrajectories(obj, generator)
    end



    function [action] = compute_action(obj, act)
        action = act/norm(act);
    end

         end

              methods(Access = private)

                  function [new_state] = detTransit(obj,state, action)

                      angle = action(:,1);
                      magnitude = action(:,2);

                      magnitude(magnitude > 2) = obj.max_action;

                      x = state(:,1);
                      y = state(:,2);

                      next_x = x + magnitude.*cosd(angle);
                      next_y = y + magnitude.*sind(angle);


                      if (next_x < obj.min_state(1) || next_x > obj.max_state(1))
                          next_x = 0;
                      end


                      if (next_y < obj.min_state(2) || next_y > obj.max_state(2))
                          next_y = 0;
                      end

                      new_state = [next_x, next_y];



                  end


              end



    end


