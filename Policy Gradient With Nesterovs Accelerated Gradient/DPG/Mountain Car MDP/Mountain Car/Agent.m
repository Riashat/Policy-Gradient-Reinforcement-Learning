classdef Agent < AgentFunctions
    
    methods(Access = public)
        
        function obj = Agent (centres, sigma, agentKernel, mdp)
            
            obj = obj @AgentFunctions (centres, sigma,agentKernel, mdp);
            
        end
        
    end
    
end

