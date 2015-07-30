load('All Results.mat')
plot([1:201], Cum_Rwd_Epsilon{1,1}, [1:201], Cum_Rwd_Epsilon{2,1},[1:201], Cum_Rwd_Epsilon{3,1},[1:201], Cum_Rwd_Epsilon{4,1})
legend('c=64','c=128','c=256','c=512')
xlabel('Number of Learning Trials')
ylabel('Cumulative Reward')
grid on
title('Natural Deterministic Policy Gradient With Adaptive Step Size (aNDPG) - Results Averaged Over 20 Experiments')