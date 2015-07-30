%function for Stochastic Policy Gradient
function [Cum_Rwd_Sigma, Step_Size_Results] = run_PGStochastic_MountainCar ()

%%set up an MDP 
noise=0.01;
gamma=0.99;
H=100;      %using H=100 for Mountain Car
Actions = 9;

%parameters for the agent
centres = 50;

%setting up the MDP here
mdp = MountainCar(noise, gamma, H, Actions);

%mdp_type_kernels = AllKernels (GridWorldKernel(mdp));
agent_kernel_type = GridWorldKernel(mdp); %make a same version of PendulumKernel - to be used with Pend MDPs

experiments = 10;
iterations = 700;
sigma = [0.4, 0.5, 0.75];  %range of sigma not needed for SPG
                    %but do this to find the optimum value of sigma for
                    %Mountain Car SPG
                    

cumulativeReward = zeros(experiments,iterations+1);
Step_Size_Results ={};
Cum_Rwd_Sigma={length(sigma)};


%range of Nesterovs mmtm values
b_step = [10, 100, 200, 500, 1000, 1500];
momentum = [0.999, 0.995, 0.99, 0.9, 0];


for s = 1:length(sigma)
    for a = 1:length(momentum)
     for b = 1:length(n_epsilon)
        for i = 1:experiments
                      
    fprintf(['\n**** EXPERIMENT NUMBER p = ', num2str(i), ' ******\n']); 
        
    agentKernel = agent_kernel_type.Kernels_State(sigma(s));     %Gaussian Kernel
    agent = Agent(centres, sigma(s), agentKernel, mdp); 
    [cum_rwd] = PGStochastic(agent, mdp, iterations, momentum(a), b_step(b), sigma(s));    
    cumulativeReward(i, :) = cum_rwd;    
        end          
    meanReward = mean(cumulativeReward(:,:));    
    Step_Size_Results{a,b} = meanReward; 
    
     end
    end
    
    Cum_Rwd_Sigma{s,:} = Step_Size_Results;

end
    %save results
    save 'All Results SPG With Nesterovs Optimization Mountain Car MDP.mat'


end
    




