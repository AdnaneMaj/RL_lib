from functools import wraps
from typing import Callable, Union, Any, Optional, List,Tuple
from inspect import signature
from models import ValueErros

class Validator:
    @staticmethod
    def _extract_param_value(func: Callable, param_name: str, self_obj, *args, **kwargs) -> Any:
        """Helper method to extract parameter value from args or kwargs."""
        sig = signature(func)
        param_names = list(sig.parameters.keys())[1:] #skip self

        #for debugging
        """
        if param_name:
            print("func",func.__name__)
            print("obj",self_obj.__class__.__name__)
            print("pram_name",param_names)
            print("kwargs",kwargs)
            print("len args)",len(args))
        """
        
        if param_name in kwargs:
            return kwargs[param_name]
        
        try:
            param_index = param_names.index(param_name)

            if param_index < len(args):
                return args[param_index]
        except ValueError:
            raise ValueError(f"Parameter {param_name} not found in function signature")
        
        return None

    @staticmethod
    def _get_attribute_or_value(obj: Any, value: Union[int, str, float]) -> Any:
        """Helper method to get either a direct value or class attribute."""
        return value if isinstance(value, (int, float)) else getattr(obj, value)

    """
    @staticmethod
    def _initialize_attributes(self_obj: Any, func: Callable, args: tuple, kwargs: dict, params_to_init: List[str]) -> None:
        #Helper method to initialize required attributes before validation.
        for param in params_to_init:
            value = Validator._extract_param_value(func, param, self_obj, *args, **kwargs)
            setattr(self_obj, param, value)
    """

    @classmethod
    def bounds(cls, min_val: Union[int,float, str], max_val: Union[int,float, str], param_name: str) -> Callable:
        """Validate that a parameter falls within specified bounds."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(self_obj, *args, **kwargs):
                value = cls._extract_param_value(func, param_name, self_obj, *args, **kwargs)
                if value is not None:
                    actual_min = cls._get_attribute_or_value(self_obj, min_val)
                    actual_max = cls._get_attribute_or_value(self_obj, max_val)
                    if not actual_min <= value <= actual_max:
                        raise ValueError(
                            f"Parameter '{param_name}' with value {value} "
                            f"must be between {actual_min} and {actual_max}"
                        )
                return func(self_obj, *args, **kwargs)
            return wrapper
        return decorator

    @classmethod
    def shape(cls, shape:Tuple[int]=None ,param_name: str = None) -> Callable:
        """Validate that a parameter has the specified shape."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(self_obj, *args, **kwargs):

                if func.__name__=="__init__" :
                    actual_shape = None
                    if self_obj.__class__.__name__ in ["MP","MRP"]:
                        actual_shape = (self_obj.n_states,self_obj.n_states)
                    if self_obj.__class__.__name__ == "MDP":
                        actual_shape = (self_obj.n_states,self_obj.n_states,self_obj.n_actions)


                value = cls._extract_param_value(func, param_name, self_obj, *args, **kwargs)
                if value is not None:
                    if actual_shape is None:
                        actual_shape = shape
                    if value.shape != actual_shape:
                        raise ValueError(
                            f"Parameter '{param_name}' with value {value} "
                            f"must be of shape {actual_shape}, got {value.shape}"
                        )
                    
                    #Initialsie:
                    if func.__name__=="__init__" and param_name=="P":
                        self_obj.P = value
                return func(self_obj, *args, **kwargs)
            return wrapper
        return decorator

    @classmethod
    def unique_list(cls, param_name: str) -> Callable:
        """Validate that a parameter is a list with unique values."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(self_obj, *args, **kwargs):
                #Get the value of the parameter to validate
                value = cls._extract_param_value(func, param_name, self_obj, *args, **kwargs)

                #Valdaite
                if value is not None:
                    if len(value) != len(set(value)):
                        raise ValueError(
                            f"Parameter '{param_name}' with value {value} "
                            f"must be a list of unique values. The provided list contains duplicates"
                        )
                    
                    #Initilaise S and n_states
                    if func.__name__=="__init__":
                        if param_name=="S":
                            self_obj.S = value
                            self_obj.n_states = len(value)
                        elif param_name=="A":
                            self_obj.A = value
                            self_obj.n_actions = len(value)
                return func(self_obj, *args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def _validate_reward(self_obj,R:dict):
        """Validate if R.keys have one of the correct 4 format :
            - Format 0 : s
            - Format 1 : (s,s_)
            - Format 2 : (s,a)
            - Format 3 : (s,s_,a)
        """
        #1.Validate format
        potential_format = list(R.keys())[0]
        potential_format_lenght = None
        format_id = None
        if isinstance(potential_format,str):
            format_id = 0
            R = {tuple([s]):v for s,v in R.items()}
        elif isinstance(potential_format,tuple) :
            potential_format_lenght = len(potential_format)
        else:
            raise ValueError(f'R.keys() must be strings or tuple, got {type(potential_format)}')

        is_valid = all(len(a) for a in list(R.keys()))
        if not is_valid:
            raise ValueError(
                f"R.keys should be in one of the following formats : s | (s),(s,s_),(s,s_,a),(s,s_)"
            )

        if format_id is None:
            if potential_format_lenght == 3:
                format_id = 3
            elif self_obj.__class__.__name__ == "MRP":
                format_id = 1
            elif self_obj.__class__.__name__ == "MDP":
                format_id = 2

        #2. Validate state and actions
        
        for r in R.keys():
            # Validate states
            if (r[0] not in self_obj.S) or (format_id in [1,3] and (len(r)>1 and r[1] not in self_obj.S)):
                raise ValueError(ValueErros.INVALID_STATES_R.value)
            if format_id in [2, 3] and ((len(r)>1 and r[1] not in self_obj.A) or (len(r) > 2 and r[2] in self_obj.A)):
                raise ValueError(ValueErros.INVALID_ACTIONS_R.value)
        return R,format_id
    

    @classmethod
    def Reward(cls, param_name: str='R',params_to_init: List[str] = None) -> Callable:
        """Validate that R is in one of those two versions, either R(s) or R(s,s') """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(self_obj, *args, **kwargs):
                R = cls._extract_param_value(func, param_name, self_obj, *args, **kwargs)
                if R:
                    R,format_id = cls._validate_reward(self_obj,R)
                    self_obj.R = R
                    self_obj.R_v = format_id #initialise RB
                return func(self_obj, *args, **kwargs)
            return wrapper
        return decorator
