function [cum_rwd] = PGStochastic(agent, mdp, iterations, momentum, b, sigma)

fprintf('\Stochastic Policy Gradient\n');

%get return of current policy
cum_rwd = zeros(1,iterations+1);
sample_traj = mdp.H;
cum_rwd(1) = mdp.estCumRwd(@agent.policy, sample_traj);
fprintf(['MDP Estimated Reward= ', num2str(cum_rwd(1)), '\n'])
mdp_type_kernels = AllKernels (GridWorldKernel(mdp));

Old_Velocity = agent.zeroVariables();

for p = 2:(iterations+1)
         
    fprintf(['\n**** Iteration p = ', num2str(p-1), ' ******\n']);
    
    %momentum schedule - dependent on number of iterations
%     mom_update = -1 - log2( (p/250) + 1);       %do not use /250 there - just use p OR 25*p
%     min_mom = 1 - 2.^mom_update;
%     max_mom = 0.995;    
%     momentum = min( min_mom, max_mom);
    
       
    current_variables = agent.variables;

    agent.update_variables(current_variables);
  
    
    %pulling trajectory/data at every iteration
    trajectory = mdp.pull_trajs( @agent.policy, 1, mdp.H);      
    [prev_state, prev_action, succ_state] = trajectory_data(trajectory);
      
    old_Traj = [prev_state prev_action];
    new_Traj = [succ_state];
                   
     expected_values = Expected_Functions_Class (mdp_type_kernels, agent, old_Traj, new_Traj);   
     rho_d = Rho_Integrator(expected_values, mdp);  
                            
     [critic] = QFunctionApproxClass(expected_values,mdp, agent, old_Traj, new_Traj); 
           
     %compute gradient with pulled data for every iteration
     [policy_gradient] = pg_gradient(agent, critic, rho_d, old_Traj, new_Traj);
            
     gradient_inc = policy_gradient';     
     %%%%%% gradient estimate per iteration until here 
    a=1;
    eta_k = a / (p + b);       %init step size = a/b; decay rate = 1+p/b
     
    %agent.update_variables(current_variables);  
    lastReward = mdp.estCumRwd(@agent.policy, sample_traj);
      
    Velocity = momentum*Old_Velocity + eta_k * gradient_inc;
       
    newVariables = current_variables + Velocity;   
    
    agent.update_variables (newVariables);        
    newReward  = mdp.estCumRwd(@agent.policy, sample_traj);
    
    Old_Velocity = Velocity;

    if newReward < lastReward
        agent.update_variables(current_variables);
        newReward = lastReward;
    end
    
          
    %reporting the reward
    estRwd = newReward;   
    fprintf(['Estimated Reward from SPG Nesterovs Optimization= ', num2str(estRwd), '\n']); 
    fprintf('Sigma= %3d | n_epsilon=%3d | momentum=%3d \n' , sigma, b, momentum);

    
    cum_rwd(p) = estRwd;    

end
end  
 









    