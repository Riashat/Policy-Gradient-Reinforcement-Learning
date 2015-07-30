classdef GaussianKernel < Kernel 
    
    properties (SetAccess = private)
        mdp;
        bwidth;
    end
    
    methods(Access = public)
        
        function obj = GaussianKernel(mdp, bwidth)
            obj.mdp = mdp;
            obj.bwidth = bwidth;
        end
        
        function K = compute_kernel(obj, X, Z)         
           sqrDistance = obj.mdp.sqDistance(X,Z);
            
           %sqrDistance = (X - Z).^2;
            
            K = exp(-(sqrDistance/ (2*(obj.bwidth^2))));
        end
        
        
        
    end
end