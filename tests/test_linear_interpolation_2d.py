import einops
import torch
import numpy as np

from torch_image_lerp import sample_image_2d, insert_into_image_2d


def test_sample_image_2d():
    # basic sanity check only
    image = torch.rand((28, 28))

    # make an arbitrary stack (..., 2) of 2d coords
    coords = torch.tensor(np.random.randint(low=0, high=27, size=(6, 7, 8, 2)))

    # sample
    samples = sample_image_2d(image=image, coordinates=coords)
    assert samples.shape == (6, 7, 8)


def test_sample_image_2d_complex_input():
    # basic sanity check only
    image = torch.complex(real=torch.rand((28, 28)), imag=torch.rand(28, 28))

    # make an arbitrary stack (..., 2) of 2d coords
    coords = torch.tensor(np.random.randint(low=0, high=27, size=(6, 7, 8, 2)))

    # sample
    samples = sample_image_2d(image=image, coordinates=coords)
    assert samples.shape == (6, 7, 8)


def test_insert_into_image_2d():
    image = torch.zeros((28, 28)).float()

    # single value
    value = torch.tensor([5]).float()
    coordinate = torch.tensor([10.5, 14.5]).view((1, 2))

    # sample
    image, weights = insert_into_image_2d(value, coordinates=coordinate, image=image)

    # check value (5) is evenly split over 4 nearest pixels
    expected = einops.repeat(torch.tensor([5 / 4]), '1 -> 2 2')
    assert torch.allclose(image[10:12, 14:16], expected)

    # check for zeros elsewhere
    assert torch.allclose(image[:10, :14], torch.zeros_like(image[:10, :14]))


def test_insert_into_image_2d_multiple():
    image = torch.zeros((28, 28)).float()

    # multiple values
    values = torch.ones(size=(6, 7, 8)).float()
    coordinates = torch.tensor(np.random.randint(low=0, high=27, size=(6, 7, 8, 2)))

    # sample
    image, weights = insert_into_image_2d(values, coordinates=coordinates, image=image)

    # check for nonzero value at one point
    sample_point = coordinates[0, 0, 0]
    y, x = sample_point
    assert image[y, x] > 0
