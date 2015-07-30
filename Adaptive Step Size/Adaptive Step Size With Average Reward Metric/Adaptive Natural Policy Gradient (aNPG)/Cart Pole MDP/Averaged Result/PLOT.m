load('Averaged Result.mat')
plot([1:1801], meanCumRwd, 'r')
grid on
legend('Adaptive Natural Deterministic Policy Gradient - Cart Pole')
xlabel('Number of Learning Trials')
ylabel('Cumulative Reward')
title('Cart Pole MDP - Adaptive Natural Deterministic Policy Gradient - Averaged Over 5 Experiments')
