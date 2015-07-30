function [cum_rwd] = PGDeterministic(agent, mdp, iterations, c_epsilon,  sigma)

fprintf('\Adaptive Natural Deterministic Policy Gradient - Cart Pole\n');

%get return of current policy
cum_rwd = zeros(1,iterations+1);
sample_traj = mdp.H;
cum_rwd(1) = mdp.estCumRwd(@agent.policy, sample_traj);
fprintf(['MDP Estimated Reward= ', num2str(cum_rwd(1)), '\n'])
mdp_type_kernels = AllKernels (GridWorldKernel(mdp));


for p = 2:(iterations+1)
         
    fprintf(['\n**** Iteration p = ', num2str(p-1), ' ******\n']); 

    current_variables = agent.variables;
    
    %pulling trajectory/data at every iteration
    trajectory = mdp.pull_trajs( @agent.policy_exploratory, 1, mdp.H);      
    [prev_state, prev_action, succ_state] = trajectory_data(trajectory);
      
    old_Traj = [prev_state prev_action];
    new_Traj = [succ_state];
                   
     expected_values = Expected_Functions_Class (mdp_type_kernels, agent, old_Traj, new_Traj);   
     rho_d = Rho_Integrator(expected_values, mdp);  
                            
     [critic] = QFunctionApproxClass(expected_values,mdp, agent, old_Traj, new_Traj); 
     
     [vanilla_gradient] = pg_gradient(agent, critic,  rho_d, old_Traj, new_Traj);
           
     [natural_gradient] = Natural_Gradient(agent, critic, old_Traj, new_Traj);
     
     gradient_vanilla = vanilla_gradient';
     
     gradient_inc = natural_gradient;
         
%      eta_k = 1 / (transpose(gradient_vanilla) * gradient_inc) ;
%      
      epsilon = c_epsilon / sqrt(p);
%      
     
     if (p==2)
         A = 1 / (transpose(gradient_vanilla) * gradient_inc) ;         
     end
     
     decayRate = (10*p)/iterations;
     
     eta_k = A / (1 + decayRate );
     
     
%      total_step = epsilon * eta_k;

    agent.update_variables(current_variables);  
    lastReward = mdp.estCumRwd(@agent.policy, sample_traj);
    
    newVariables = current_variables +  epsilon * eta_k * gradient_inc;
    
    
    agent.update_variables (newVariables);        
    newReward  = mdp.estCumRwd(@agent.policy, sample_traj);

    if newReward < lastReward
        agent.update_variables(current_variables);
        newReward = lastReward;
    end
    
          
    %reporting the reward
    estRwd = newReward;   
    fprintf(['Estimated Reward from Natural Deterministic Policy Gradient= ', num2str(estRwd), '\n']); 
    fprintf('Sigma= %3d | c = %3d | step_size = %3d \n' , sigma,  c_epsilon, eta_k);    

    cum_rwd(p) = estRwd;    

end
end  
 









    