%function for Stochastic Policy Gradient
function [Cum_Rwd_Sigma, Step_Size_Results] = run_PGDeterministic_Pendulum ()

%%set up an MDP 
noise=0;
gamma=0.99;
H=400;
Actions = 9;

%parameters for the agent
centres = 200;

%setting up the MDP here
mdp = Pendulum(noise, gamma, H, Actions);

%mdp_type_kernels = AllKernels (GridWorldKernel(mdp));
agent_kernel_type = GridWorldKernel(mdp); %make a same version of PendulumKernel - to be used with Pend MDPs

%Gaussian Kernel 
experiments = 15;
iterations = 400;

sigma = [0.1:0.2:1];

cumulativeReward = zeros(experiments,iterations+1);
Step_Size_Results ={};
Cum_Rwd_Sigma={length(sigma)};

%a=1; b=100; %learning rate parameters - found manually
%for testing

a_param = 2.^[0:1:20];
b_param = 2.^[0:1:20];


for s = 1:length(sigma)
    for a = 1:length(a_param)
     for b = 1:length(b_param)
        for i = 1:experiments  
            
    fprintf(['\n**** EXPERIMENT NUMBER p = ', num2str(i), ' ******\n']); 

    agentKernel = agent_kernel_type.Kernels_State(sigma(s));     %Gaussian Kernel
    agent = Agent(centres, sigma(s), agentKernel, mdp); 
    [cum_rwd] = PGDeterministic(agent, mdp, iterations, a_param(a), b_param(b), sigma(s));    
    cumulativeReward(i, :) = cum_rwd;    
 
        end   
        
    meanReward = mean(cumulativeReward(:,:));
    
    Step_Size_Results{a,b} = meanReward;
    
     end
    end   
    
    Cum_Rwd_Sigma{s,:} = Step_Size_Results;
end

    %save results
    save 'All Results DPG Pendulum MDP.mat'


end
    




