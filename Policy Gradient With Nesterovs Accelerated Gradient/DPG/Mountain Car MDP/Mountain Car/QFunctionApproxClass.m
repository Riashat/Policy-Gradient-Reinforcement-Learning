
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
        function [q_weight_approx] = QfunctionApprox (obj, state, action)            
       q_weight_approx= QFunctionLeastSquares (obj.mdp, obj.agent, obj.expected_values, state, action);
            
        end


        
        
        function Q = getQValue(obj,state, action)
        Q = obj.expected_values.featuremap(state, action) * obj.QfunctionApprox(state, action);         
        end
        
    

        end


    end






