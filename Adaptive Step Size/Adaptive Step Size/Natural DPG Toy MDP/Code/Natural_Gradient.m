function [gradient_value, R] = Natural_Gradient(agent, critic, old_traj, new_traj)
      
      gradient_value = critic.QfunctionApprox();
     
      successor_state = new_traj;
      successor_action = agent.policy(successor_state);
      
      Q = critic.getQValue(successor_state, successor_action);
     
      R = max(Q);



end

