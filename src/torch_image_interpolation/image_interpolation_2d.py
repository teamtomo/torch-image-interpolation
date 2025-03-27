from typing import Literal

import einops
import torch
import torch.nn.functional as F

from torch_image_interpolation import utils
from .grid_sample_utils import array_to_grid_sample


def sample_image_2d(
    image: torch.Tensor,
    coordinates: torch.Tensor,
    interpolation: Literal['nearest', 'bilinear', 'bicubic'] = 'bilinear',
) -> torch.Tensor:
    """Sample a 2D image with a specific interpolation mode.

    Parameters
    ----------
    image: torch.Tensor
        `(h, w)` image or `(c, h, w)` multi-channel image.
    coordinates: torch.Tensor
        `(..., 2)` array of coordinates at which `image` should be sampled.
        - Coordinates are ordered `yx` and are positions in the `h` and `w` dimensions respectively.
        - Coordinates span the range `[0, N-1]` for a dimension of length N.
    interpolation: Literal['nearest', 'bilinear', 'bicubic']
        Interpolation mode for image sampling.

    Returns
    -------
    samples: torch.Tensor
        `(..., )` or `(..., c)` array of samples from `image`.
    """
    device = coordinates.device

    if image.ndim not in (2, 3):
        raise ValueError(f'image should have shape (h, w) or (c, h, w), got {image.shape}')

    # keep track of a few image properties
    input_image_is_complex = torch.is_complex(image)
    input_image_is_multichannel = image.ndim == 3

    # coerce single channel to multi-channel
    if input_image_is_multichannel is False:
        image = einops.rearrange(image, 'h w -> 1 h w')

    # set up for sampling with torch.nn.functional.grid_sample
    # shape (..., 2) -> (n, 2)
    coordinates, ps = einops.pack([coordinates], pattern='* yx')
    n_samples = coordinates.shape[0]
    h, w = image.shape[-2:]

    # handle complex input
    if input_image_is_complex:
        # cannot sample complex tensors directly with grid_sample
        # c.f. https://github.com/pytorch/pytorch/issues/67634
        # workaround: treat real and imaginary parts as separate channels
        image = torch.view_as_real(image)
        image = einops.rearrange(image, 'c h w complex -> (complex c) h w')

    # torch.nn.functional.grid_sample is set up for sampling grids
    # here we view our image as a batch of n_samples multi-channel images
    # then sample a batch of (1x1) grids
    # this enables sampling arbitrarily shaped arrays of coords
    image = einops.repeat(image, 'c h w -> b c h w', b=n_samples)
    coordinates = einops.rearrange(coordinates, 'b yx -> b 1 1 yx')  # b h w yx

    # take the samples
    samples = F.grid_sample(
        input=image,
        grid=array_to_grid_sample(coordinates, array_shape=image.shape[-2:]),
        mode=interpolation,
        padding_mode='border',  # this increases sampling fidelity at edges
        align_corners=True,
    )

    # reconstruct complex valued samples if required
    if input_image_is_complex is True:
        samples = einops.rearrange(samples, 'b (complex c) 1 1 -> b c complex', complex=2)
        samples = utils.view_as_complex(samples.contiguous())  # (b, c)
    else:
        samples = einops.rearrange(samples, 'b c 1 1 -> b c')

    # set samples from outside of image to zero explicitly
    coordinates = einops.rearrange(coordinates, 'b 1 1 yx -> b yx')
    image_shape = torch.as_tensor(image.shape[-2:]).to(device)
    inside = torch.logical_and(coordinates >= 0, coordinates <= image_shape - 1)
    inside = torch.all(inside, dim=-1)  # (b,)
    samples[~inside] *= 0

    # pack samples back into the expected shape
    # (b, c) -> (..., c)
    [samples] = einops.unpack(samples, pattern='* c', packed_shapes=ps)

    # ensure output has correct shape
    # - (...) if input image was single channel
    # - (..., c) if input image was multi-channel
    if input_image_is_multichannel is False:
        samples = einops.rearrange(samples, '... 1 -> ...')  # drop channel dim

    return samples


def insert_into_image_2d(
    values: torch.Tensor,
    coordinates: torch.Tensor,
    image: torch.Tensor,
    weights: torch.Tensor | None = None,
    interpolation: Literal['nearest', 'bilinear'] = 'bilinear',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert values into a 2D image with bilinear interpolation.

    Parameters
    ----------
    values: torch.Tensor
        `(...)` or `(..., c)` array of values to be inserted into `image`.
    coordinates: torch.Tensor
        `(..., 2)` array of 2D coordinates for each value in `data`.
        - Coordinates are ordered `yx` and are positions in the `h` and `w` dimensions respectively.
        - Coordinates span the range `[0, N-1]` for a dimension of length N.
    image: torch.Tensor
        `(h, w)` or `(c, h, w)` array containing the image into which data will be inserted.
    weights: torch.Tensor | None
        `(h, w)` array containing weights associated with each pixel in `image`.
        This is useful for tracking weights across multiple calls to this function.
    interpolation: Literal['nearest', 'bilinear']
        Interpolation mode used for adding data points to grid.

    Returns
    -------
    image, weights: tuple[torch.Tensor, torch.Tensor]
        The image and weights after updating with data from `data` at `coordinates`.
    """
    # keep track of a few properties of the inputs
    input_image_is_multichannel = image.ndim == 3
    h, w = image.shape[-2:]

    # validate inputs
    values_shape = values.shape[:-1] if input_image_is_multichannel else values.shape
    coordinates_shape, coordinates_ndim = coordinates.shape[:-1], coordinates.shape[-1]

    if values_shape != coordinates_shape:
        raise ValueError('One coordinate pair is required for each value in data.')
    if coordinates_ndim != 2:
        raise ValueError('Coordinates must be 2D with shape (..., 2).')
    if weights is None:
        weights = torch.zeros(size=(h, w), dtype=torch.float32, device=image.device)

    # add channel dim to both image and values if input image is not multichannel
    if not input_image_is_multichannel:
        image = einops.rearrange(image, 'h w -> 1 h w')
        values = einops.rearrange(values, '... -> ... 1')

    # linearise data and coordinates
    values, _ = einops.pack([values], pattern='* c')
    coordinates, _ = einops.pack([coordinates], pattern='* yx')
    coordinates = coordinates.float()

    # only keep data and coordinates inside the image
    upper_bound = torch.tensor(image.shape, device=image.device) - 1
    idx_inside = (coordinates >= 0) & (coordinates <= upper_bound)
    idx_inside = torch.all(idx_inside, dim=-1)
    values, coordinates = values[idx_inside], coordinates[idx_inside]

    # splat data onto grid
    if interpolation == 'nearest':
        image = _insert_nearest_2d(values, coordinates, image, weights)
    if interpolation == 'bilinear':
        image, weights = _insert_linear_2d(values, coordinates, image, weights)
    return image, weights


def _insert_nearest_2d(
    data,  # (b, c)
    coordinates,  # (b, yx)
    image,  # (c, h, w)
    weights  # (h, w)
):
    # b is number of samples per channel, c is number of channels
    b, c = data.shape

    # flatten data to insert values for all channels with one call to _index_put()
    data = einops.rearrange(data, 'b c -> (b c)')

    # repeat yx coords for insertion into each of c channels
    coordinates = torch.round(coordinates).long()
    coordinates = einops.repeat(coordinates, 'b yx -> (b c) yx', c=c)

    # grab indices for insertion in each spatial dimension for all channels
    idx_y, idx_x = einops.rearrange(coordinates, '(b c) yx -> yx (b c)')

    # now get corresponding channel indices
    idx_c = torch.arange(c, device=coordinates.device, dtype=torch.long)
    idx_c = einops.repeat(idx_c, 'c -> (b c)', b=b)

    # insert image data into all channels
    image.index_put_(indices=(idx_c, idx_y, idx_x), values=data, accumulate=True)

    # insert ones into weights image at each position
    w = torch.ones(len(coordinates), device=weights.device, dtype=weights.dtype)
    weights.index_put_(indices=(idx_y, idx_x), values=w, accumulate=True)
    return image


def _insert_linear_2d(data, coordinates, image, weights):
    # calculate and cache floor and ceil of coordinates for each value to be inserted
    corner_coords = torch.empty(size=(data.shape[0], 2, 2), dtype=torch.long, device=image.device)
    corner_coords[:, 0] = torch.floor(coordinates)
    corner_coords[:, 1] = torch.ceil(coordinates)

    # calculate linear interpolation weights for each data point being inserted
    _weights = torch.empty(size=(data.shape[0], 2, 2), device=image.device)  # (b, 2, yx)
    _weights[:, 1] = coordinates - corner_coords[:, 0]  # upper corner weights
    _weights[:, 0] = 1 - _weights[:, 1]  # lower corner weights

    # define function for adding weighted data at nearest 4 pixels to each coordinate
    # make sure to do atomic adds, don't just override existing data at each position
    def add_data_at_corner(y: Literal[0, 1], x: Literal[0, 1]):
        w = einops.reduce(_weights[:, [y, x], [0, 1]], 'b yx -> b', reduction='prod')
        idx_y, idx_x = einops.rearrange(corner_coords[:, [y, x], [0, 1]], 'b yx -> yx b')
        image.index_put_(indices=(idx_y, idx_x), values=w * data, accumulate=True)
        weights.index_put_(indices=(idx_y, idx_x), values=w, accumulate=True)

    # insert correctly weighted data at each of 4 nearest pixels then return
    add_data_at_corner(0, 0)
    add_data_at_corner(0, 1)
    add_data_at_corner(1, 0)
    add_data_at_corner(1, 1)
    return image, weights
