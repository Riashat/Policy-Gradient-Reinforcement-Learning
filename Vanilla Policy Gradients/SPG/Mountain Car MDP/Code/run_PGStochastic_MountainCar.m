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

sigma = [0.3, 0.5, 0.7];

cumulativeReward = zeros(experiments,iterations+1);
Step_Size_Results ={};
Cum_Rwd_Sigma={length(sigma)};

 a_param = [1, 10, 25, 50, 100, 250, 500]; 
 b_param = [10, 50, 75, 100, 200, 300, 1000, 1500];


for s = 1:length(sigma)
    for a = 1:length(a_param)
     for b = 1:length(b_param)
        for i = 1:experiments 
            
    fprintf(['\n**** EXPERIMENT NUMBER p = ', num2str(i), ' ******\n']); 

    agentKernel = agent_kernel_type.Kernels_State(sigma(s));     %Gaussian Kernel
    agent = Agent(centres, sigma(s), agentKernel, mdp); 
    [cum_rwd] = PGStochastic(agent, mdp, iterations, a_param(a), b_param(b), sigma(s));    
    cumulativeReward(i, :) = cum_rwd;    
    save 'Each Experiment Result.mat'
 
        end   
    meanReward = mean(cumulativeReward(:,:));
    
    Step_Size_Results{a,b} = meanReward;
    
     end
         
     save 'Average Results for current Step-Size params.mat'

    end
    
    Cum_Rwd_Sigma{s,:} = Step_Size_Results;    
     save 'Current Sigma Results.mat'

end

    %save results
    save 'All Results SPG Mountain Car MDP.mat'


end
    




