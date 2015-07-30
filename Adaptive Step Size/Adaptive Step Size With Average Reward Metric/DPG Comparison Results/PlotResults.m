load('Results With Blobs.mat')
plot([1:201], aDPG_c_256, 'k', [1:201], aDPG_c_512, 'k',  [1:201], aNDPG_c_256, 'r', [1:201], aNDPG_c_512, 'r', [1:201], aNestDPG_c_256, 'b', [1:201], aNestDPG_c_512, 'b',  [1:201], rwdBlob1,'o', [1:201], rwdBlob2, 'o', [1:201], rwdBlob3, 'o', [1:201], totalReward,'*')

legend('Vanilla DPG', 'Vanilla DPG', 'Natural DPG', 'Natural DPG', 'Classical DPG', 'Classical DPG', 'Local Blob 1', 'Local Blob 2', 'Local Blob 3', 'Optimal Reward')

xlabel('Number of Learning Trials')
ylabel('Cumulative Reward')
title('Comparsion of Deterministic Policy Gradients - Using Adaptive Step Size')
grid on
