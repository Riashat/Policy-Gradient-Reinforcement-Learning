load('Plot Results.mat')
plot([1:201], Cum_Rwd_Epsilon{1,1},[1:201], Cum_Rwd_Epsilon{2,1},[1:201], Cum_Rwd_Epsilon{3,1})
legend('c=128', 'c=256', 'c=512')
xlabel('Number of Learning Trials')
ylabel('Cumulative Reward')
title('Adaptive Step Size in Vanilla Deterministic PG - Variation With Epsilon = c/ sqrt(p)')
grid on