
function [gradient_value] = pg_gradient(agent, critic, rho_d, old_traj, new_traj)


for i =1 
 
    rho_d.SetRho(agent);

     
     %%%%%% check state successor and action successor values in rho_integrator
     
      Prod = Product_Of_Functions (@critic.getQValue, @agent.DlogPiDTheta);
      
      gradient_value = rho_d.Integration(@Prod.multiplyFunctions);
     


end

end












    
    