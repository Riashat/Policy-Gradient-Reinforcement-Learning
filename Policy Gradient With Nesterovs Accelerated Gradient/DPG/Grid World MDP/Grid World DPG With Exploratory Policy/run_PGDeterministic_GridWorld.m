function [totalReward, Step_Size_Results, Cum_Rwd_Sigma] = run_PGDeterministic_GridWorld ()

mdp = ContinuousGridWorldMDP(0, 0.99, 50, 9)
state1 = mdp.getStartState;
state2 = mdp.transit(state1, 1);
state3 = mdp.transit(state2,1);
state4 = mdp.transit(state3,1);
state5 = mdp.transit(state4,3);
state6 = mdp.transit(state5,3);
state7 = mdp.transit(state6,3);
totalReward = mdp.reward(state1,1)+ mdp.reward(state2,1)+ mdp.reward(state3,1)+ mdp.reward(state4,1)+mdp.reward(state5,1)+mdp.reward(state6,1)+mdp.reward(state7,1)+mdp.reward(state7,1)*43

%%set up an MDP 
noise=0;
gamma=0.99;
H=50;
Actions = 9;

%parameters for the agent
centres = 25;

%setting up the MDP here
mdp = ContinuousGridWorldMDP(noise, gamma, H, Actions);

%mdp_type_kernels = AllKernels (GridWorldKernel(mdp));
agent_kernel_type = GridWorldKernel(mdp); %make a same version of PendulumKernel - to be used with Pend MDPs

experiments = 20;
iterations = 500;

sigma = [0:0.05:0.75];

cumulativeReward = zeros(experiments,iterations+1);
Step_Size_Results ={};
Cum_Rwd_Sigma={length(sigma)};

%a=1; b=100; %learning rate parameters - found manually
%for testing

%range of Nesterovs mmtm values
b_step = [10, 50, 100, 150, 300, 500, 1000, 1200];
momentum = [0.999, 0.995, 0.99, 0.9, 0];

for s = 1:length(sigma)
    for a = 1:length(momentum)
     for b = 1:length(b_step)
        for i = 1:experiments
                      
    fprintf(['\n**** EXPERIMENT NUMBER p = ', num2str(i), ' ******\n']); 

    agentKernel = agent_kernel_type.Kernels_State(sigma(s));     %Gaussian Kernel
    agent = Agent(centres, sigma(s), agentKernel, mdp); 
    [cum_rwd] = PGDeterministic(agent, mdp, iterations, momentum(a), b_step(b), sigma(s));    
    cumulativeReward(i, :) = cum_rwd;  
        save 'Each Experiment Result.mat'
 
 
        end   
    meanReward = mean(cumulativeReward(:,:));
    Step_Size_Results{a,b} = meanReward;
        save 'Each Step Size Result.mat'
    
     end
    end
    Cum_Rwd_Sigma{s,:} = Step_Size_Results;
      save 'Each Sigma Result.mat'

end
    %save results
    save 'All Results DPG Nesterovs Grid World MDP.mat'


end
    
