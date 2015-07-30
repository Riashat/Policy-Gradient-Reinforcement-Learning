%function for Deterministic Policy Gradient - Cart Pole MDP
function [opt_reward, Step_Size_Results, Cum_Rwd_Sigma] = run_PGDeterministic_CartPole ()

opt_reward = 52;

%setting up the MDP here
mdp = CartPole();

%mdp_type_kernels = AllKernels (GridWorldKernel(mdp));
agent_kernel_type = GridWorldKernel(mdp); %make a same version of PendulumKernel - to be used with Pend MDPs

%parameters for the agent
centres = 100;

experiments = 10;
iterations = 1200;
cumulativeReward = zeros(experiments,iterations+1);

sigma = [0:0.1:0.7];

Step_Size_Results ={};
Cum_Rwd_Sigma={length(sigma)};

%range of Nesterovs mmtm values
%a =1
n_epsilon = [1, 10, 100, 250, 500, 1000, 1500];
momentum = [0.999, 0.995, 0.99, 0.9, 0];

for s = 1:length(sigma)
    for a = 1:length(momentum)
     for b = 1:length(n_epsilon)
        for i = 1:experiments
                      
    fprintf(['\n**** EXPERIMENT NUMBER p = ', num2str(i), ' ******\n']); 

    agentKernel = agent_kernel_type.Kernels_State(sigma(s)); 
    agent = Agent(centres, sigma(s), agentKernel, mdp); 
    [cum_rwd] = PGDeterministic(agent, mdp, iterations, momentum(a), n_epsilon(b), sigma(s));    
    cumulativeReward(i, :) = cum_rwd;    
    save 'Each Experiment Result.mat'

 
        end   
    meanReward = mean(cumulativeReward(:,:));
    Step_Size_Results{a,b} = meanReward;
    
        save 'Step Size Results.mat'
     end
    end
    Cum_Rwd_Sigma{s,:} = Step_Size_Results;
       save 'Each Sigma Results.mat'

end
    %save results
    save 'All Results DPG Nesterovs Method Cart Pole MDP.mat'


end
    




