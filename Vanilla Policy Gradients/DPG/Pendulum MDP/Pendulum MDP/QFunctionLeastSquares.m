function [q_weight] = QFunctionLeastSquares(mdp, agent,expected_values, states_actions, succ_states)


succ_actions = agent.policy(succ_states);


%temportal difference learning for Q function
[q_weight] = TD(succ_actions,succ_states, mdp, expected_values);

end

