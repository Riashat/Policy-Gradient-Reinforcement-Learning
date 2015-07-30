%function for Stochastic Policy Gradient
function [opt_rwd, Step_Size_Results, Cum_Rwd_Sigma] = run_PGStochastic_CartPole ()

opt_rwd = 52;


%setting up the MDP here
mdp = CartPole();

%mdp_type_kernels = AllKernels (GridWorldKernel(mdp));
agent_kernel_type = GridWorldKernel(mdp); %make a same version of PendulumKernel - to be used with Pend MDPs

%parameters for the agent
centres = 100;

experiments = 10;
iterations = 1200;
cumulativeReward = zeros(experiments,iterations+1);

sigma = [ 0.4, 0.5, 0.75];   %is this the ideal range of sigma value for SPG?

Step_Size_Results ={};
Cum_Rwd_Sigma={length(sigma)};

 a_param = [1, 10, 50, 100, 200]; 
 b_param = [10, 50, 100, 150, 200, 300, 500, 1000, 1500];
 

for s = 1:length(sigma)
    for a = 1:length(a_param)
     for b = 1:length(b_param)
        for i = 1:experiments
                      
    fprintf(['\n**** EXPERIMENT NUMBER p = ', num2str(i), ' ******\n']); 
        
    agentKernel = agent_kernel_type.Kernels_State(sigma(s));     %Gaussian Kernel
    agent = Agent(centres, sigma(s), agentKernel, mdp); 
    [cum_rwd] = PGStochastic(agent, mdp, iterations, a_param(a), b_param(b), sigma(s));    
    cumulativeReward(i, :) = cum_rwd;    
        end          
    meanReward = mean(cumulativeReward(:,:));    
    Step_Size_Results{a,b} = meanReward; 
    save 'Each Experiment Result.mat'
     end
         save 'Step Size Results.mat'
    end
    
    Cum_Rwd_Sigma{s,:} = Step_Size_Results;
    
        save 'Each Sigma Results.mat'

end
    %save results
    save 'All Results SPG Cart Pole MDP.mat'


end
    




