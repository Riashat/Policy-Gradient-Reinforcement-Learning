classdef Kernel < handle
   
    methods(Access = public, Abstract)
        compute_kernel(obj, X, Z);
    end
    
end