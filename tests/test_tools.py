from imageapply.tools import apply_model, pad_to_multiple, crop_to_original
import numpy as np
import torch

def gen_id(max_batch):
    def id(x):
        if max_batch is not None and x.shape[0] > max_batch:
            raise ValueError("Batch size too large")
        return x
    return id

def run_apply_model(data):
    assert np.array_equal(apply_model(gen_id(None), data), data)
    assert np.array_equal(apply_model(gen_id(1), data, batch_size=1), data)
    assert np.array_equal(apply_model(gen_id(100), data, batch_size=100), data)
    assert np.array_equal(apply_model(gen_id(101), data, batch_size=101), data)
    assert np.array_equal(apply_model(gen_id(3), data, batch_size=3), data)

def test_apply_model_numpy():
    data = np.random.rand(100, 10, 10, 3)
    run_apply_model(data)

def test_apply_model_torch():
    data = torch.rand((100, 10, 10, 3))
    run_apply_model(data)

def run_pad_to_multiple(data, input_shape, expected_shape):
    assert np.array_equal(pad_to_multiple(data, input_shape).shape, expected_shape)
    assert np.array_equal(pad_to_multiple(data, input_shape, pad_position="centre").shape, expected_shape)

def test_pad_to_multiple_numpy():
    data = np.random.rand(1, 5, 10, 3)
    run_pad_to_multiple(data, (4, 6, 1), (1, 8, 12, 3))

def test_pad_to_multiple_torch():
    data = torch.rand((1, 5, 10, 3))
    run_pad_to_multiple(data, (4, 6, 1), (1, 8, 12, 3))

def test_crop_to_original_shape():
    shape = np.random.randint(1, 100, 4)
    data = np.random.rand(*shape)
    assert np.array_equal(crop_to_original(pad_to_multiple(data, (4, 6, 1)), shape).shape, shape)
    assert np.array_equal(crop_to_original(pad_to_multiple(data, (4, 6, 1), pad_position="centre"), shape).shape, shape)

def test_crop_to_original():
    data = np.random.rand(1, 5, 10, 3)
    assert np.array_equal(crop_to_original(pad_to_multiple(data, (4, 6, 1)), (1, 5, 10, 3)), data)
    assert np.array_equal(crop_to_original(pad_to_multiple(data, (4, 6, 1), pad_position="centre"), (1, 5, 10, 3), pad_position="centre"), data)