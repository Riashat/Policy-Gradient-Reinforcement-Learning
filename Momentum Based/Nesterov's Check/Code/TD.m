function [TD_Approximation_W] = TD(succ_actions ,succ_states, mdp, expected_values)

Psi = expected_values.featuremapQ(succ_states, succ_actions);
Mult_P_PsiX = Psi(2:end, :);

%incorrect here -  MultPPsi should have same dimension as Psi
Mult_P_Psi = [Mult_P_PsiX; ones(1,25)];

idx = length(Psi(1,:));

LS_gamma = 0.98; %  %check if this is a reasonable value or not
%regularizer = 0.00001;

regularizer = 0.0001;

reward_on_traj = mdp.reward(succ_states, succ_actions);


%add regularization term to avoid matrix becoming singular
% LSTD gamma should be 0.98
% Psi and subtracted (Psi * Transition Probability - alternatve to using
% the expected function?

A_vec = (Psi'* (Psi - LS_gamma*Mult_P_Psi) + regularizer*eye(idx, idx));
b = (Psi' * reward_on_traj);




TD_Approximation_W =  A_vec \ b ;
    

end

