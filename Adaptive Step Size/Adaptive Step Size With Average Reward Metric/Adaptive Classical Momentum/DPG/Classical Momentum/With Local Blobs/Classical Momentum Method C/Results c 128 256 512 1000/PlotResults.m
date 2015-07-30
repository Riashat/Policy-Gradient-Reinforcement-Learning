plot([1:201], Step_Size_Results{1,1},[1:201], Step_Size_Results{1,2},[1:201], Step_Size_Results{1,3},[1:201], Step_Size_Results{1,4})
legend('M=0.999, c=128', 'M=0.999, c=256', 'M=0.999, c=512', 'M=0.999, c=1000')
xlabel('Number of Learning Trials')
ylabel('Cumulative Reward')
title('Adaptive Policy Gradient With Nesteros Accelerated Gradient')
grid on