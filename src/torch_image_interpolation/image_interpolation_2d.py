from typing import Literal

import einops
import torch
import torch.nn.functional as F

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
        `(h, w)` image.
    coordinates: torch.Tensor
        `(..., 2)` array of coordinates at which `image` should be sampled.
        - Coordinates are ordered `yx` and are positions in the `h` and `w` dimensions respectively.
        - Coordinates span the range `[0, N-1]` for a dimension of length N.
    interpolation: Literal['nearest', 'bilinear', 'bicubic']
        Interpolation mode for image sampling.

    Returns
    -------
    samples: torch.Tensor
        `(..., )` array of samples from `image`.
    """
    if len(image.shape) != 2:
        raise ValueError(f'image should have shape (h, w), got {image.shape}')

    # set up for sampling with torch.nn.functional.grid_sample
    # shape (..., 2) -> (n, 2)
    coordinates, ps = einops.pack([coordinates], pattern='* yx')
    n_samples = coordinates.shape[0]
    h, w = image.shape[-2:]

    # handle complex input
    complex_input = torch.is_complex(image)
    if complex_input is True:
        # cannot sample complex tensors directly with grid_sample
        # c.f. https://github.com/pytorch/pytorch/issues/67634
        # workaround: treat real and imaginary parts as separate channels and sample independently
        image = torch.view_as_real(image)
        image = einops.rearrange(image, 'h w complex -> complex h w')
        image = einops.repeat(image, 'complex h w -> b complex h w', b=n_samples)
        coordinates = einops.rearrange(coordinates, 'b zyx -> b 1 1 zyx')  # b h w zyx
    else:
        image = einops.repeat(image, 'h w -> b 1 h w', b=n_samples)  # b c h w
        coordinates = einops.rearrange(coordinates, 'b zyx -> b 1 1 zyx')  # b h w zyx

    # take the samples at each point
    samples = F.grid_sample(
        input=image,
        grid=array_to_grid_sample(coordinates, array_shape=(h, w)),
        mode=interpolation,
        padding_mode='border' if interpolation == 'bilinear' else 'reflection',
        # this increases sampling fidelity at edges
        align_corners=True,
    )

    # if input was complex, reconstruct complex numbers from channels
    if complex_input is True:
        samples = einops.rearrange(samples, 'b complex 1 1 -> b complex')
        samples = torch.view_as_complex(samples.contiguous())  # (b, )
    else:
        samples = einops.rearrange(samples, 'b 1 1 1 -> b')

    # set samples from outside of image to zero explicitly
    coordinates = einops.rearrange(coordinates, 'b 1 1 yx -> b yx')
    image_shape = torch.as_tensor((h, w), device=image.device)
    inside = torch.logical_and(coordinates >= 0, coordinates <= image_shape - 1)
    inside = torch.all(inside, dim=-1)  # (b, d, h, w)
    samples[~inside] *= 0

    # pack samples back into the expected shape and return
    # shape (n, ) -> (...)
    [samples] = einops.unpack(samples, pattern='*', packed_shapes=ps)
    return samples  # (...)


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
        `(...)` array of values to be inserted into `image`.
    coordinates: torch.Tensor
        `(..., 2)` array of 2D coordinates for each value in `data`.
        - Coordinates are ordered `yx` and are positions in the `h` and `w` dimensions respectively.
        - Coordinates span the range `[0, N-1]` for a dimension of length N.
    image: torch.Tensor
        `(h, w)` array containing the image into which data will be inserted.
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
    if values.shape != coordinates.shape[:-1]:
        raise ValueError('One coordinate pair is required for each value in data.')
    if coordinates.shape[-1] != 2:
        raise ValueError('Coordinates must be of shape (..., 2).')
    if weights is None:
        weights = torch.zeros_like(image)

    # linearise data and coordinates
    values, _ = einops.pack([values], pattern='*')
    coordinates, _ = einops.pack([coordinates], pattern='* zyx')
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


def _insert_nearest_2d(data, coordinates, image, weights):
    coordinates = torch.round(coordinates).long()
    idx_y, idx_x = einops.rearrange(coordinates, 'b yx -> yx b')
    image.index_put_(indices=(idx_y, idx_x), values=data, accumulate=False)
    w = torch.ones(len(coordinates), device=weights.device, dtype=weights.dtype)
    weights.index_put_(indices=(idx_y, idx_x), values=w, accumulate=True)
    return image


def _insert_linear_2d(data, coordinates, image, weights):
    # calculate and cache floor and ceil of coordinates for each value to be inserted
    corner_coords = torch.empty(size=(data.shape[0], 2, 2), dtype=torch.long, device=image.device)
    corner_coords[:, 0] = torch.floor(coordinates)
    corner_coords[:, 1] = torch.ceil(coordinates)

    # calculate linear interpolation weights for each data point being inserted
    _weights = torch.empty(size=(data.shape[0], 2, 2), device=image.device
                           )  # (b, 2, yx)
    _weights[:, 1] = coordinates - corner_coords[:, 0]  # upper corner weights
    _weights[:, 0] = 1 - _weights[:, 1]  # lower corner weights

    # define function for adding weighted data at nearest 4 pixels to each coordinate
    # make sure to do atomic adds, don't just override existing data at each position
    def add_data_at_corner(y: Literal[0, 1], x: Literal[0, 1]):
        w = einops.reduce(_weights[:, [y, x], [0, 1]], 'b yx -> b', reduction='prod')
        idx_y, idx_x = einops.rearrange(corner_coords[:, [y, x], [0, 1]],'b yx -> yx b')
        image.index_put_(indices=(idx_y, idx_x), values=w * data, accumulate=True)
        weights.index_put_(indices=(idx_y, idx_x), values=w, accumulate=True)

    # insert correctly weighted data at each of 4 nearest pixels then return
    add_data_at_corner(0, 0)
    add_data_at_corner(0, 1)
    add_data_at_corner(1, 0)
    add_data_at_corner(1, 1)
    return image, weights
