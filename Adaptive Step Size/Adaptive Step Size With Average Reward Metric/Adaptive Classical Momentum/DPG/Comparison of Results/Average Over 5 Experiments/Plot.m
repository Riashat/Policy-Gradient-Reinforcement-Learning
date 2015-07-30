plot([1:201], CM_DPG_c_512, 'r', [1:201], DPG_Nesterov_ConstantNu_c_512, 'k', [1:201], DPG_Nesterov_Decay_c_512, 'b', [1:201], totalReward, '*')
legend('Classical Momentum', 'Nesterov With Constant Nu', 'Nesterov With Decaying Nu=1/iterations')
xlabel('Number of Learning Trials')
ylabel('Cumulative Reward')
title('DPG Comparison of Momentum Gradient Descent With Adaptive Step Size')
grid on