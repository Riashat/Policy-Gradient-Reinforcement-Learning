classdef ContinuousGridWorldMDP < MDP 
        
     properties(SetAccess = private)
         
     end
     
     methods(Access = public)
         
         function obj = ContinuousGridWorldMDP(noise, gamma, H, actionDiscCount)
            actionDiscCount = 9;
            H=50;
            noise =  0.02;
            gamma = 0.99;           
            %********************
                
            start_state_noise = [0.1 0.1]; %0.1 if you want a bit of initial noise in the system
            state_dim = 2;
            action_dim = 1;
            max_action = 5;
            %for discretizing
            min_disc_act = -max_action;
            max_disc_act = max_action;
            %for sampling
            max_state = [4 4];
            min_state = [0 0];
            
            name = 'ContinuousGridWorldMDP';
            spatial_dim = 1;
                      
            %***************************            
            obj = obj@MDP(gamma, H, noise, start_state_noise, state_dim, action_dim, max_action,...
                min_disc_act, max_disc_act, max_state, min_state, actionDiscCount, name, spatial_dim);           
            obj.actionDiscCount = actionDiscCount;
            %obj.all_actions = obj.discretizeActions();
            obj.discAct = 0; %continuous        
         end
         
         function [new_state, rwd] = engine(obj,state, action)
            [new_state] = obj.transit(state, action);
            [rwd] = obj.reward(state,action);

         end
        
          function [rwd] = reward(obj,state,actions)
  
          goal_state = obj.goal_state();          
          costs_goal = sum((bsxfun(@minus, goal_state(:,:), state(:,:))).^2, 2);         
          goal =  exp(-  costs_goal/2);
         
          alpha = 0.4;
          cost1 = sum((bsxfun(@minus, [1 1], state(:,:))).^2, 2); 
          rwd1 =  alpha.*exp(-  cost1/2);
          
          rwd = rwd1 + goal;
                  
         end
                                
         
         function [goal_state] = goal_state(obj)       
          goal_state = [3 3];
         end
         
         
         function [start_state] = getStartState(obj)
   
             start_state = obj.start_state_noise*randn(1);
             %start_state = [0 0];
         end
         
            function [state] = uniform_state(obj)
                state1 = unifrnd(obj.min_state(1), obj.max_state(1));
                state2 = unifrnd(obj.min_state(2), obj.max_state(2));
                state = [state1 state2];
            end
            
            
             function [new_state] = transit(obj,state, action)
                %Stochastic Transition
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
  %*****************************      
     end     
              
         
          methods(Access = private)
              
              function [new_state] = detTransit(obj,state, action)
                %Deterministic Transition
                x = state(:,1);
                y = state(:,2);
                
                    if action>=0 && action <=1
                        new_state = [x, y+1];    
                    else
                        if action> 1 && action <=2
                            new_state = [x, y-1];
                    else
                        if action> 2 && action <=3
                            new_state = [x+1, y];
                    else 
                        if action> 3 && action <=4
                            new_state = [x-1, y];     
                    else
                        if action> 4 && action <=5
                            new_state = [x, y];
                        else 
                            if action > obj.max_action || action < 0
                                new_state = [x, y];
                            end
                        end
                        end
                        end
                        end
                    end                       
                next_x = new_state(:,1);
                next_y = new_state(:,2);
                
                  if (next_x < obj.min_state(1) || next_x > obj.max_state(1) || next_y < obj.min_state(2) || next_y > obj.max_state(2))
                    new_state = [0 0];
                  end
                  

              end
                
          end
          
end



