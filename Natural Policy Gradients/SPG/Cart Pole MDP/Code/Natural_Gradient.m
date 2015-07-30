function [gradient_value] = Natural_Gradient(agent, critic, old_traj, new_traj)
      
      gradient_value = critic.QfunctionApprox();
     


end

