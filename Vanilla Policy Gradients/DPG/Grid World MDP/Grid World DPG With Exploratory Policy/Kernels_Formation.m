classdef Kernels_Formation < Kernel
    
    % Evaluates the original kernel in blocks to save memory
    
    properties (SetAccess = private)
        obj_kernels;
     
    end
    
    methods
        function obj = Kernels_Formation(obj_kernels)
            obj.obj_kernels = obj_kernels;
            
        end
        
        function [K] = Compute_Kernel(obj, X, Z)
         
        end
        
    end
end