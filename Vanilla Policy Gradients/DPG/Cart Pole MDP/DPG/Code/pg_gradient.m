
function [gradient_value] = pg_gradient(agent, critic, rho_d, old_traj, new_traj)


for i =1 
 
    rho_d.SetRho(agent);

     
    
      Prod = Product_Of_Functions (@critic.getQValue, @agent.GradQAction);
      
      gradient_value = rho_d.Integration(@Prod.multiplyFunctions);
     


end

end












    
    