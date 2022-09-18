from abc import abstractmethod
from typing import Callable, List, Tuple, Union
from imageapply.tools import pad_to_multiple, crop_to_original, divide_into_regions, combine_regions
from imageapply.data import T
import numpy as np

class CombinedModel:
    """
    A model with all combined operations. Can be used to sandwich multiple ReversibleOperations around a base model.
    """
    
    def __init__(self, steps: List[Callable]):
        self.steps = steps
        
        model = self.steps[-1]
        for step in reversed(self.steps[:-1]):
            if step is not None:
                model = step(model)
        self.model = model
        
    def __call__(self, data:T) -> T:
        return self.model(data)
    

class ExtendibleOperation(Callable):
    """
    An operation that can be extended with a sub-operation.
    """
    
    def __init__(self):
        self._sub = lambda x: x
    
    def _check_and_update(self, sub: Union[T, Callable]):
        if isinstance(sub, Callable):
            self._update(sub)
            return True
        return False
    
    def _update(self, sub: Callable):
        self._sub = sub
    
    @abstractmethod
    def __call__(self, data_or_sub: Union[T, Callable]) -> Union[T, Callable]:
        pass


class BatchOperation(ExtendibleOperation):
    """
    Performs a number of sub-operations on the same data and combines the result.
    """
    
    def __init__(self, sub_operations: List[ExtendibleOperation], combine: Callable=None):
        super().__init__()
        self._sub = sub_operations
        self.combine = combine
        
    def _check_and_update(self, sub: Union[T, Callable]):
        if isinstance(sub, Callable):
            for s in self._sub:
                if isinstance(s, ExtendibleOperation):
                    s._update(sub)
            return True
        return False
    
    def __call__(self, data_or_sub: Union[T, Callable]) -> Union[T, Callable]:
        if self._check_and_update(data_or_sub):
            return self
        return self.combine([sub(data_or_sub) for sub in self._sub])
        
        
class ReversibleOperation(ExtendibleOperation):
    """
    A data operation that can be reversed.
    """
    
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _forward(self, data:T) -> T:
        raise NotImplementedError("forward not implemented")
    
    @abstractmethod
    def _backward(self, data:T) -> T:
        raise NotImplementedError("backward not implemented")
    
    def __call__(self, data_or_inter:Union[T, Callable]) -> T:
        if self._check_and_update(data_or_inter):
            return self
        return self._backward(self._sub(self._forward(data_or_inter)))

class Flip(ReversibleOperation):
    """
    Flips the Data along an axis.
    """
    
    def __init__(self, axis:Union[int, Tuple[int]]):
        super().__init__()
        self.axis = axis
        
    def _forward(self, data:T) -> T:
        return np.flip(data, self.axis)
    
    def _backward(self, data:T) -> T:
        return np.flip(data, self.axis)

    def __repr__(self):
        return f"Flip(axis={self.axis})"

class Id(ReversibleOperation):
    """
    Id operation.
    """
    
    def __init__(self):
        super().__init__()
        
    def _forward(self, data:T) -> T:
        return data

    def _backward(self, data: T) -> T:
        return data
    
    def __repr__(self) -> str:
        return f"Id()"
    
class BasicTTA(BatchOperation):
    # Temporary test
    
    def __init__(self, combine_mode:str="mean"):
        if combine_mode == "mean":
            super().__init__(sub_operations=[Id(), Flip(1), Flip(2), Flip((1,2))], combine=lambda x: np.mean(x, axis=0))
        else:
            raise ValueError(f"Combine mode {combine_mode} not supported")
        self.combine_mode = combine_mode
    
    def __repr__(self):
        return f"BasicTTA(combine_mode={self.combine_mode})"

# class BasicTTA(ReversibleOperation):
#     """
#     Basic Test Time Augmentation. Flips image horizontally, vertically and both.
#     """
    
#     def __init__(self, combine_mode:str="mean"):
#         super().__init__(distributed=True)
#         self.combine_mode = combine_mode
    
#     def _forward(self, data):
#         return [
#             data,
#             np.flip(data, 1),
#             np.flip(data, 2),
#             np.flip(data, (1, 2))
#         ]
    
#     def _backward(self, data):
#         data = [
#             data[0],
#             np.flip(data[1], 1),
#             np.flip(data[2], 2),
#             np.flip(data[3], (1, 2))
#         ]
        
#         if self.combine_mode == 'mean':
#             return np.mean(data, axis=0)
#         else:
#             raise ValueError(f"Unknown combine mode {self.combine_mode}")
        
#     def __repr__(self) -> str:
#         return f"BasicTTA(combine_mode={self.combine_mode})"


class PadCrop(ReversibleOperation):
    """
    Pad the data to a multiple of the input size, reverse crops the data to the original shape.
    """
    def __init__(self, input_size, pad_mode="reflect", pad_position="end"):
        super().__init__()
        self.input_size = input_size
        self.pad_mode = pad_mode
        self.pad_position = pad_position
    
    def _forward(self, data):
        self.original_shape = data.shape
        return pad_to_multiple(data, self.input_size, pad_mode=self.pad_mode, pad_position=self.pad_position)
    
    def _backward(self, data):
        return crop_to_original(data, self.original_shape, pad_position=self.pad_position)
    
    def __repr__(self):
        return f"PadCrop(input_size={self.input_size}, pad_mode={self.pad_mode}, pad_position={self.pad_position})"
    
class DivideCombine(ReversibleOperation):
    """
    Divide the data into regions of size region_size, reverse combines the regions into the original shape.
    """
    def __init__(self, region_size:Tuple):
        """
        Args:
            region_size: The size of the regions to divide the data into. 
        """
        super().__init__()
        self.region_size = region_size
    
    def _forward(self, data):
        """
        Forward pass, divide the data into regions.
        """
        self.original_shape = data.shape # save the original shape for the backward pass
        return divide_into_regions(data, self.region_size)
    
    def _backward(self, data):
        return combine_regions(data, self.region_size, self.original_shape)
    
    def __repr__(self):
        return f"DivideCombine(region_size={self.region_size})"
    