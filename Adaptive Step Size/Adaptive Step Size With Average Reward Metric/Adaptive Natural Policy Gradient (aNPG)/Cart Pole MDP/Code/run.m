function [meanReward, Cum_Rwd_Sigma, cumulativeReward, cum_rwd ] = run()

opt_reward = 52;

%setting up the MDP here
mdp = CartPole();

%mdp_type_kernels = AllKernels (GridWorldKernel(mdp));
agent_kernel_type = GridWorldKernel(mdp); %make a same version of PendulumKernel - to be used with Pend MDPs

%parameters for the agent
centres = 100;

experiments = 1;
iterations = 1800;

cumulativeReward = zeros(experiments,iterations+1);

sigma = 0.5;

Cum_Rwd_Sigma={length(sigma)};

%c_epsilon = 512;

%c_epsilon = 1500;


c_epsilon = 2500;

for s = 1:length(sigma)

    for e = 1:length(c_epsilon)
    
        for i = 1:experiments
                      
    fprintf(['\n**** EXPERIMENT NUMBER p = ', num2str(i), ' ******\n']); 

    agentKernel = agent_kernel_type.Kernels_State(sigma(s));     %Gaussian Kernel
    agent = Agent(centres, sigma(s), agentKernel, mdp); 
    [cum_rwd] = PGDeterministic(agent, mdp, iterations, c_epsilon(e), sigma(s));    
    cumulativeReward(i, :) = cum_rwd;    
 
        end
        
     meanReward = mean(cumulativeReward(:,:));
      
     Cum_Rwd_Epsilon{e,:} = meanReward;
    end
       
     Cum_Rwd_Sigma{s,:} = Cum_Rwd_Epsilon;

end
    %save results
    save 'All Results Cart Pole Adaptive NDPG.mat'

    
end

    




