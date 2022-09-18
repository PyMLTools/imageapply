#TODO: Improve Package Name
#TODO: Add support for single image calls (i.e. without batch dimension)
#TODO: Add support for non-batched models (models that can only handle one image at a time)
#TODO: Add support for pytorch tensors (and others)
#TODO: Limit image dims to 3
#TODO: Implement a Data class (with Pytorch, Numpy, and Tensorflow subclasses) to handle the different data types

from .revtransform import PadCrop, DivideCombine, CombinedModel

class FlexibleModel:
    """
    This model is designed to be used with models that can only handle a certain input size.
    """

    def __init__(self, model, input_size, max_batch_size=None, basic_tta=False):
        """
        Creates a new FlexibleModel object.
        
        Args:
            model (function): The model to apply to the data
            input_size (tuple): The size of the input to the model
            max_batch_size (int): The maximum batch size to use when applying the model
            basic_tta (bool): Whether to use basic test time augmentation
            
        Returns:
            FlexibleModel: The new FlexibleModel object
        """
        assert model is not None and callable(model), "Model must be callable"
        assert input_size[0] is None, "First dimension of input size must be None"

        self.model = model
        self.input_size = input_size[1:]
        self.max_batch_size = max_batch_size
        self.tta = basic_tta
        
        self.combined = CombinedModel([
            PadCrop(self.input_size, pad_mode="zeros", pad_position="end"),
            DivideCombine(self.input_size),
            model
        ]) 
    
    def __call__(self, batch):
        """
        Runs the model on the batch of data.
        
        Args:
            batch (T): The batch of data to run the model on
        
        Returns:
            T: The output of the model
        """
        # Assume for now, data is a batch numpy array
        return self.combined(batch) 