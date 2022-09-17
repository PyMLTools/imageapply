# Image Apply
![test](https://github.com/PyMLTools/imageapply/actions/workflows/tests.yml/badge.svg)

A package for improving the simplicity of adapting Segmentation Models to large images.

This package include a class that can be used to apply a segmentation model to large images. The class will split the image into smaller tiles, apply the model to each tile, and then stitch the tiles back together. This allows the model to be applied to images that are too large to fit in memory.

## Installation

```bash
pip install imageapply
```

## Usage
This package can be used to easily extend a segmentation model to large images. The following example is for a `model` that takes a batch of images of size (None, 256, 256, 3) as input and outputs a mask of size (None, 256, 256, 1).

```python
from imageapply import FlexibleModel
model = FlexibleModel(model, input_size=(256, 256, 3), max_batch_size=16)

# Apply the model to a batch of large images of size (20, 1024, 1024, 3)
masks = model(images)
``` 
