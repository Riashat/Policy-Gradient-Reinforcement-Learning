function [q_weight] = QFunctionLeastSquares_2(mdp, agent,expected_values, state, action)

%succ_actions = agent.policy(succ_states);

%temportal difference learning for Q function
[q_weight] = TD(action,state, mdp, expected_values);

end

