from abc import abstractmethod
from typing import Callable, List, Tuple
from .tools import pad_to_multiple, crop_to_original, divide_into_regions, combine_regions
from .data import T

class CombinedModel:
    """
    A model with all combined transformations. Can be used to sandwich multiple ReversibleTransformations around a base model.
    """
    
    def __init__(self, steps: List[Callable]):
        self.steps = steps
        
        model = self.steps[-1]
        for step in reversed(self.steps[:-1]):
            model = step(model)
        self.model = model
        
    def __call__(self, data:T) -> T:
        return self.model(data)
        

class ReversibleTransformation:
    """
    A data transformation that can be reversed.
    """
    
    def __init__(self):
        self._inter = id

    def _attach_intermediary(self, intermediary:Callable[[T], T]):
        self._inter = intermediary
    
    @abstractmethod
    def _forward(self, data):
        raise NotImplementedError("forward not implemented")
    
    @abstractmethod
    def _backward(self, data):
        raise NotImplementedError("backward not implemented")
    
    def __call__(self, data_or_inter:T|Callable[[T], T]) -> T:
        if isinstance(data_or_inter, Callable):
            self._attach_intermediary(data_or_inter)
            return self
        else:
            return self._backward(self._inter(self._forward(data_or_inter)))


class PadCrop(ReversibleTransformation):
    """
    Pad the data to a multiple of the input size, reverse crops the data to the original shape.
    """
    def __init__(self, input_size, pad_mode="reflect", pad_position="end"):
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
    
class DivideCombine(ReversibleTransformation):
    """
    Divide the data into regions of size region_size, reverse combines the regions into the original shape.
    """
    def __init__(self, region_size:Tuple):
        """
        Args:
            region_size: The size of the regions to divide the data into. 
        """
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