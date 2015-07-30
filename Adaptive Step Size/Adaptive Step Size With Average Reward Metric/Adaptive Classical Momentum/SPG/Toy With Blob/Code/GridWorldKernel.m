classdef GridWorldKernel < Kernel_Storage
    
    properties (Access=private, Constant)
        
    end
    
    properties (SetAccess = private)
        mdp;
        
    end
    
    methods (Access = public)
        function obj = GridWorldKernel (mdp)
            obj.mdp = mdp;
        end
        
     
        function new_kernels = Kernels_State(obj, sigma)       
            new_kernels = GaussianKernel(obj.mdp, sigma);
        end
        
        %implement more kernels here relevant to the GridWorld
        
        
    end
    
end

