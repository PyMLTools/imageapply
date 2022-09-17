#TODO: Improve Package Name
#TODO: Add support for single image calls (i.e. without batch dimension)
#TODO: Add support for non-batched models (models that can only handle one image at a time)
#TODO: Add support for pytorch tensors (and others)
#TODO: Limit image dims to 3
#TODO: Implement a Data class (with Pytorch, Numpy, and Tensorflow subclasses) to handle the different data types

import numpy as np
from .tools import apply_model, pad_to_multiple, crop_to_original, divide_into_regions, combine_regions

class FlexibleModel:

    def __init__(self, model, input_size, max_batch_size=None):
        assert model is not None and callable(model), "Model must be callable"
        assert input_size[0] is None, "First dimension of input size must be None"

        self.model = model
        self.input_size = input_size[1:]
        self.max_batch_size = max_batch_size
    
    def __call__(self, batch):
        # Assume for now, data is a batch numpy array
        return self._pad_and_apply(batch)

    def _pad_and_apply(self, batch, pad_mode="zeros", pad_position="end"):
        out = pad_to_multiple(batch, self.input_size, pad_mode=pad_mode, pad_position=pad_position)
        out = self._apply_on_size_multiple(out)
        out = crop_to_original(out, batch.shape, pad_position=pad_position)
        return out

    def _apply_on_size_multiple(self, batch):
        # First dim is batch dim

        out = divide_into_regions(batch, self.input_size)        

        out = apply_model(self.model, out, batch_size=self.max_batch_size)

        out = combine_regions(out, self.input_size, batch.shape)

        return out
