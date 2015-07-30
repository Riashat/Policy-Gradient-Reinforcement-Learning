function [expected_functions_class] = Expected_Functions_Class(mdp_type_kernels, agent, old_traj, new_traj)

        expected_functions_class = Expectation(mdp_type_kernels, agent);
        
        expected_functions_class.exp_func(old_traj, new_traj);
        
end


