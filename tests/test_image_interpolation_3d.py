import numpy as np
import torch
import einops

from torch_image_interpolation import sample_image_3d, insert_into_image_3d


def test_sample_image_3d():
    # basic sanity check only
    image = torch.rand((28, 28, 28))

    # make an arbitrary stack (..., 3) of 3d coords
    coords = torch.tensor(np.random.randint(low=0, high=27, size=(6, 7, 8, 3)))

    # sample
    samples = sample_image_3d(image=image, coordinates=coords)
    assert samples.shape == (6, 7, 8)


def test_sample_image_3d_complex_input():
    # basic sanity check only
    image = torch.complex(real=torch.rand((28, 28, 28)), imag=torch.rand(28, 28, 28))

    # make an arbitrary stack (..., 3) of 3d coords
    coords = torch.tensor(np.random.randint(low=0, high=27, size=(6, 7, 8, 3)))

    # sample
    samples = sample_image_3d(image=image, coordinates=coords)
    assert samples.shape == (6, 7, 8)


def test_insert_into_image_3d():
    image = torch.zeros((28, 28, 28)).float()

    # single value
    value = torch.tensor([5]).float()
    coordinate = torch.tensor([10.5, 14.5, 18.5]).view((1, 3))

    # sample
    image, weights = insert_into_image_3d(value, coordinates=coordinate, image=image, interpolation="trilinear")

    # check value (5) is evenly split over 4 nearest pixels
    expected = einops.repeat(torch.tensor([5 / 8]), '1 -> 2 2 2')
    assert torch.allclose(image[10:12, 14:16, 18:20], expected)

    # check for zeros elsewhere
    assert torch.allclose(image[:10, :14, :18], torch.zeros_like(image[:10, :14, :18]))


def test_insert_into_image_3d_multiple():
    image = torch.zeros((28, 28, 28)).float()

    # multiple values
    values = torch.ones(size=(6, 7, 8)).float()
    coordinates = torch.tensor(np.random.randint(low=0, high=27, size=(6, 7, 8, 3)))

    # sample
    image, weights = insert_into_image_3d(values, coordinates=coordinates, image=image, interpolation="trilinear")

    # check for nonzero value at one point
    sample_point = coordinates[0, 0, 0]
    z, y, x = sample_point
    assert image[z, y, x] > 0


def test_insert_into_image_nearest_interp_3d():
    image = torch.zeros((28, 28, 28)).float()

    # single value
    value = torch.tensor([5]).float()
    coordinate = torch.tensor([10.7, 14.3, 18.9]).view((1, 3))

    # sample
    image, weights = insert_into_image_3d(value, coordinates=coordinate, image=image, interpolation='nearest')

    # check value (5) is added at nearest pixel
    expected = torch.zeros((28, 28, 28)).float()
    expected[11, 14, 19] = 5
    assert torch.allclose(image, expected)