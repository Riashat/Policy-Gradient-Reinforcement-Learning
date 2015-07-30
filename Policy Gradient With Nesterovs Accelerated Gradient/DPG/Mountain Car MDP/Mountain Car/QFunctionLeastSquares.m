function [q_weight] = QFunctionLeastSquares(mdp, agent,expected_values, state,action)
[q_weight] = TD(action,state, mdp, expected_values);
end

