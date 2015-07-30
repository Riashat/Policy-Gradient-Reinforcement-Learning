classdef Expectation < Expected_Functions
    
    properties (Access = private)
      
    end
    
    methods (Access = public)
        function obj = Expectation (mdp_type_kernels, agent)
            obj = obj @Expected_Functions (mdp_type_kernels, agent);
            
        end
        
    end
       

end

