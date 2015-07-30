classdef Agent < AgentFunctions
    
    methods(Access = public)
        
        function obj = Agent (initialized_states, sigma, agentKernel, mdp)
            
            obj = obj @AgentFunctions (initialized_states, sigma,agentKernel, mdp);
            
        end
        
    end
    
end

