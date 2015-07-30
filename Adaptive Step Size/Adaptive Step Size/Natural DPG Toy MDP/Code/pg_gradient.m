
function [gradient_value, R] = pg_gradient(agent, critic, rho_d, old_traj, new_traj)


for i =1 
 
    rho_d.SetRho(agent);

     
    
      Prod = Product_Of_Functions (@critic.getQValue, @agent.GradQAction);
      
      gradient_value = rho_d.Integration(@Prod.multiplyFunctions);
     
      
      successor_state = new_traj;
      successor_action = agent.policy(successor_state);
      
      Q = critic.getQValue(successor_state, successor_action);
     
      R = max(Q);



end

end












    
    