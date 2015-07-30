    classdef Expected_Functions < handle

        properties (SetAccess = protected)

        old_TRAJ = [];
        new_TRAJ = [];
        expected_states;
        sigma;

        end

        properties (SetAccess = private)

         mdp_type_kernels;
         agent;
         

        end


        methods (Access = public)

            function obj = Expected_Functions ( mdp_type_kernels, agent)

                obj.mdp_type_kernels = mdp_type_kernels; 
                obj.agent = agent;
                

            end



            function conditional_function (obj, state)
                obj.expected_states = [];

                %same_state = zeros(length(state(:,1)),1);

                length_state = length(state(:,1));

                new_expected_state = [];
                new_exp_state = [];

                for i=1:(ceil(length_state))
                    new_expected_state = obj.calculateState_Conditioned_Previous(state(i,:));
                    
                    %new_exp_state = sparse([new_exp_state; new_expected_state]);
                    
                end

                obj.expected_states = new_expected_state;
            end



            function s_conditioned_old_sa = calculateState_Conditioned_Previous(obj, state)

                
                % this is not correct
                %this should be the same as the density function in the MDP
                s_conditioned_old_sa = exp(- (state - obj.old_TRAJ).^2) / (2*(obj.sigma)^2);

            end


                function expectation_value = conditional_expectation(obj,Value,x)

                    if(exist('x'))
                        expectation_value = obj.TransitionProbability(x) * Value;
                    else

                    expectation_value = obj.TransitionProbability () * Value;
                    end

                end



                function transition = TransitionProbability (obj, state)
                        if (exist('state'))
                            obj.conditional_function(state);
                        end

                        transition = obj.expected_states;

                end

                
                function exp_func (obj, oldTraj, newTraj)
                    
                    obj.old_TRAJ = oldTraj;
                    obj.new_TRAJ = newTraj;
                    
                end
                
                

                %k determines the order of polynomial for the feature map
                
                %this function is defined separately in a function,
                %and doesnt get used through Expected_Functions class
                
                
                
                 function featureMap = featuremap(obj, state, action)
                     
                     featureMap = obj.agent.GradQAction(state, action);

                 end
                 
                 function [fmQ] = featuremapQ(obj, state, action)
                     fmQ = obj.agent.PhiSA(state, action);
                 end

        end

    end












