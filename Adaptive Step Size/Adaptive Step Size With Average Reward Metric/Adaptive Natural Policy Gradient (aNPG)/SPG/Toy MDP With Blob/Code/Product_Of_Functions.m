classdef Product_Of_Functions < FunctionClass
    

    properties (SetAccess = private)
       func_hand_1
        func_hand_2;
    end
    
    methods(Access = public)
        
        function obj = Product_Of_Functions(func_hand_1,func_hand_2)
            obj.func_hand_1 = func_hand_1;
            obj.func_hand_2 = func_hand_2;
        end
        
        function mult = multiplyFunctions(obj, varargin)
            mult = bsxfun(@times,obj.func_hand_1(varargin{:}),obj.func_hand_2(varargin{:}));
          
        end
        
    end
end