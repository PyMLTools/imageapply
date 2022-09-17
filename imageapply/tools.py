import numpy as np
import torch
from typing import Callable, TypeVar
from .data import data_object

T = TypeVar("T", np.ndarray, torch.Tensor)

def apply_model(model : Callable[[T], T], data : T, batch_size=None) -> T:
    """
    Applies the model to a batch of data, running at most batch_size entries through at once. 
    """

    if batch_size is None or batch_size > data.shape[0]:
        return model(data)

    assert type(batch_size) is int and batch_size > 0, "Batch size must be a positive integer"

    results = []
    for i in range(0, data.shape[0], batch_size):
        results.append(model(data[i:i+min(batch_size, data.shape[0]-i)]))

    return np.concatenate(results, axis=0)

def divide_into_regions(data : T, region_size : tuple) -> T:

    # Assumes data is a multiple of region_size
    # data is array of shape (Batch_size, ...)
    # region_size is tuple of length data.ndim - 1

    n_splits = [data.shape[i+1] // region_size[i] for i in range(len(region_size))]
    for dim in range(len(region_size)):
        data = np.concatenate(np.split(data, n_splits[dim], axis=dim+1), axis=0)
    
    return data

def combine_regions(data : T, region_size : tuple, original_size : tuple) -> T:
    """
    Combines the regions of the data into the original shape.

        Parameters:
            data (T): The data to combine
            region_size (tuple): The size of the regions
            original_size (tuple): The original shape of the data

        Returns:
            T: The combined data
    """
    n_splits = [original_size[i+1] // region_size[i] for i in range(len(region_size))]
    for dim in range(len(region_size)-1, -1, -1):
        data = np.concatenate(np.split(data, n_splits[dim], axis=0), axis=dim+1)

    return data

####################################################################################################
###################################### PADDING AND CROPPING ########################################
####################################################################################################

def pad_to_multiple(data:T, 
                    input_size:tuple, 
                    pad_mode:str="zeros", 
                    pad_position:str="end") -> T:
    """
    Pads the data to be a multiple of the input size.

        Parameters: 
            data (data_object): The data to pad
            input_size (tuple): The size of the model input. Pads to the smallest multiple of input_size \
                that is greater than the data size
            pad_mode (str): The mode to use for padding. One of "zeros", "reflect", "symmetric", "edge", "wrap"
            pad_position (str): The position to pad. One of "end" or "centre"
        
        Returns:
            data_object: The padded data
    """

    rem_per_dim = [data.shape[i+1] % input_size[i] for i in range(len(input_size))]
    pad_amount = [input_size[i] - rem_per_dim[i] if rem_per_dim[i] > 0 else 0 for i in range(len(input_size))]
    if pad_position == "end":
        pad_amount = [(0, 0)] + [(0, pad_amount[i]) for i in range(len(pad_amount))]
    elif pad_position == "centre":
        pad_amount = [(0, 0)] + [(pad_amount[i]//2, pad_amount[i] - pad_amount[i]//2) for i in range(len(pad_amount))]
    else:
        raise ValueError("Invalid pad position. Must be 'end' or 'centre'")

    if pad_mode == "zeros":
        return np.pad(data, pad_amount, mode='constant', constant_values=0)
    elif pad_mode == "reflect":
        return np.pad(data, pad_amount, mode='reflect')
    elif pad_mode == "symmetric":
        return np.pad(data, pad_amount, mode='symmetric')
    else:
        raise ValueError("Invalid pad mode. Must be 'zeros', 'reflect', or 'symmetric'")

def crop_to_original(data : data_object, original_shape : tuple, pad_position : str = "end") -> data_object:
    """
    Crops the data to the original shape.
    
        Parameters:
            data (data_object): The data to crop
            original_shape (tuple): The original shape of the data
            pad_position (str): The position to pad. One of "end" or "centre"
            
        Returns:
            data_object: The cropped data
    """
    if pad_position == "end":
        crop_indices = tuple([slice(None)] + [slice(0, original_shape[i+1]) for i in range(len(original_shape)-1)])
    elif pad_position == "centre":
        pad_amount = [data.shape[i+1] - original_shape[i+1] for i in range(len(original_shape)-1)]
        crop_indices = tuple([slice(None)] + [slice(pad_amount[i]//2, pad_amount[i]//2 + original_shape[i+1]) for i in range(len(original_shape)-1)])
    return data[crop_indices]