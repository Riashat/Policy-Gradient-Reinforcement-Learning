plot([1:201], Step_Size_Results{1,1}, 'k',[1:201], Step_Size_Results{1,2}, 'r',[1:201], Step_Size_Results{1,3}, 'b', [1:201], Step_Size_Results{1,4}, 'green')
legend('Epsilon Parameter c=128', 'Epsilon Parameter c=256', 'Epsilon Parameter c=512', 'Epsilon Parameter c=1000')
xlabel('Number of Learning Trials')
ylabel('Cumulative Reward')
title('Adaptive Classical Momentum - SPG On Toy MDP With Variations of Epsilon Parameter c')
grid on