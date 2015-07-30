classdef AllKernels < Kernel_Storage
    
     properties (Access = private)
        obj_kernels;
        
     end
     
       methods (Access = public)
        function obj = AllKernels(obj_kernels)
            obj.obj_kernels = obj_kernels;
        
        end
        
         function new_kernels = Kernels_State(obj, sigma) %  kernels for phi(s)        
            new_kernels = Kernels_Formation (obj.obj_kernels.Kernels_State(bwidth));
            
        end
        
       end
    
end


    