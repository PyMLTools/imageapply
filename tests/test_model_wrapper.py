from imageapply import FlexibleModel
import numpy as np

#TODO: Add a lot more tests

def id(x):
    return x

def almost_id(x):
    x[:, 0, 0, :] = 0
    return x

def test_flexible_model():
    data = np.random.rand(100, 10, 10, 3)
    model = FlexibleModel(id, (None, 5, 2, 1))

    assert np.array_equal(model(data), data)

def test_flexible_model_padding():
    data = np.random.rand(1, 5, 10, 3)
    model = FlexibleModel(id, (None, 4, 6, 1))

    assert np.array_equal(model(data), data)
    
def test_flexible_model_tta():
    data = np.random.rand(1, 5, 10, 3)
    model = FlexibleModel(id, (None, 4, 6, 1), basic_tta=True)
    
    assert np.array_equal(model(data), data)
    
    data = np.zeros((1, 2, 2, 1))
    data[0, -1, -1, 0] = 1
    
    # For the horizontal and vertical flipped version, almost_id will set the first pixel to 0
    # Resulting in a 0.75 mean between the 4 different flips
    model = FlexibleModel(almost_id, (None, 4, 6, 1), basic_tta=True)
    assert model(data)[0, -1, -1, 0] == 0.75