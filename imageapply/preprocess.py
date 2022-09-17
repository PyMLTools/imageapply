# Preprocessing tools for training
from data import data_object

def divide_into_regions(data : data_object, region_size : tuple, pad_mode : str) -> data_object:
    """
    Divides the data into regions of size region_size.
    """

    assert len(region_size) == len(data.shape) - 1, "Region size must have same number of dimensions as data"

