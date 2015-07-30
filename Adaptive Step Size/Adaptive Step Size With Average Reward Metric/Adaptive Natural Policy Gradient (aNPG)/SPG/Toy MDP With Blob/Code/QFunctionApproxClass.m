
    classdef QFunctionApproxClass  < QMainSuperClass

        properties (SetAccess = private)
            QApprox;
            %feature_map
            expected_values;
            q_weight;
            mdp;
            agent
            state_actions;
            succ_states;
        end


        methods (Access=public)

            %function obj = QFunctionApproxClass(expected_values,featuremap, mdp)
            function obj = QFunctionApproxClass(expected_values,mdp, agent, state_actions, succ_states)
               
                obj.expected_values = expected_values;
                obj.mdp = mdp;
                obj.agent = agent;
                obj.state_actions = state_actions;
                obj.succ_states = succ_states;
                %obj.q_weight = q_weight;
            end

        %this function must return the parameters w of the linear function
        %approximator       
       function [q_weight_approx] = QfunctionApprox (obj)              
       %q_weight_approx= QFunctionLeastSquares (obj.mdp, obj.agent, obj.expected_values, state, action);   
       q_weight_approx= QFunctionLeastSquares (obj.mdp, obj.agent, obj.expected_values, obj.state_actions, obj.succ_states);
       end  
       
       function [q_weight_approx] = QfunctionApprox_2 (obj, state, action)              
       q_weight_approx= QFunctionLeastSquares_2 (obj.mdp, obj.agent, obj.expected_values, state, action);   
       %q_weight_approx= QFunctionLeastSquares (obj.mdp, obj.agent, obj.expected_values, obj.state_actions, obj.succ_states);
       end  
       
       
         
        function Q = getQValue(obj,state, action)
        % Q - GradLogPiPolicy * w - get w from QfunctionApprox 
        Q = obj.expected_values.featuremap(state,action) * obj.QfunctionApprox_2(state, action);
           
        end

        end


    end






