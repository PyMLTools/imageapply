from imageapply import FlexibleModel
import numpy as np

#TODO: Add a lot more tests

def id(x):
    return x

def test_flexible_model():
    data = np.random.rand(100, 10, 10, 3)
    model = FlexibleModel(id, (None, 5, 2, 1))

    assert np.array_equal(model(data), data)

def test_flexible_model_padding():
    data = np.random.rand(1, 5, 10, 3)
    model = FlexibleModel(id, (None, 4, 6, 1))

    assert np.array_equal(model(data), data)