classdef ProductFunctionClass < handle
   
    methods(Access = public, Abstract)
        multiplyFunctions(obj, x);
    end
    
end