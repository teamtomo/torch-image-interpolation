from typing import Literal

import einops
import torch
import torch.nn.functional as F

from .grid_sample_utils import array_to_grid_sample


def sample_image_3d(
    image: torch.Tensor,
    coordinates: torch.Tensor,
    interpolation: Literal['nearest', 'trilinear'] = 'trilinear',
) -> torch.Tensor:
    """Sample a 3D image with a specific interpolation mode.

    Parameters
    ----------
    image: torch.Tensor
        `(d, h, w)` image or `(c, d, h, w)` multi-channel image.
    coordinates: torch.Tensor
        `(..., 3)` array of coordinates at which `image` should be sampled.
        - Coordinates are ordered `zyx` and are positions in the `d`, `h` and `w` dimensions respectively.
        - Coordinates span the range `[0, N-1]` for a dimension of length N.
    interpolation: Literal['nearest', 'trilinear']
        Interpolation mode for image sampling.

    Returns
    -------
    samples: torch.Tensor
        `(..., )` or `(..., c)` array of samples from `image`.
    """
    device = coordinates.device

    if image.ndim not in (3, 4):
        raise ValueError(f'image should have shape (d, h, w) or (c, d, h, w), got {image.shape}')

    # keep track of a few image properties
    input_image_is_complex = torch.is_complex(image)
    input_image_is_multichannel = image.ndim == 4

    # coerce single channel to multi-channel
    if input_image_is_multichannel is False:
        image = einops.rearrange(image, 'd h w -> 1 d h w')

    # setup coordinates for sampling image with torch.nn.functional.grid_sample
    # shape (..., 3) -> (b, 3)
    coordinates, ps = einops.pack([coordinates], pattern='* zyx')
    n_samples = coordinates.shape[0]

    # handle complex input
    if input_image_is_complex:
        # cannot sample complex tensors directly with grid_sample
        # c.f. https://github.com/pytorch/pytorch/issues/67634
        # workaround: treat real and imaginary parts as separate channels
        image = torch.view_as_real(image)
        image = einops.rearrange(image, 'c d h w complex -> (complex c) d h w')

    # torch.nn.functional.grid_sample is set up for sampling grids
    # here we view our volume as a batch of n_samples multi-channel volumes
    # then sample a batch of (1x1x1) grids
    # this enables sampling arbitrarily shaped arrays of coords
    image = einops.repeat(image, 'c d h w -> b c d h w', b=n_samples)
    coordinates = einops.rearrange(coordinates, 'b zyx -> b 1 1 1 zyx')  # b d h w zyx

    # take the samples
    interpolation = 'bilinear' if interpolation == 'trilinear' else interpolation
    samples = F.grid_sample(
        input=image,
        grid=array_to_grid_sample(coordinates, array_shape=image.shape[-3:]),
        mode=interpolation,  # bilinear is trilinear sampling when input is volumetric
        padding_mode='border',  # this increases sampling fidelity at edges
        align_corners=True,
    )

    # reconstruct complex valued samples if required
    if input_image_is_complex is True:
        samples = einops.rearrange(samples, 'b (complex c) 1 1 1 -> b c complex', complex=2)
        samples = torch.view_as_complex(samples.contiguous())  # (b, c)
    else:
        samples = einops.rearrange(samples, 'b c 1 1 1 -> b c')

    # set samples from outside of volume to zero
    coordinates = einops.rearrange(coordinates, 'b 1 1 1 zyx -> b zyx')
    volume_shape = torch.as_tensor(image.shape[-3:]).to(device)
    inside = torch.logical_and(coordinates >= 0, coordinates <= volume_shape - 1)
    inside = torch.all(inside, dim=-1)  # (b, d, h, w)
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


def insert_into_image_3d(
    values: torch.Tensor,
    coordinates: torch.Tensor,
    image: torch.Tensor,
    weights: torch.Tensor | None = None,
    interpolation: Literal['nearest', 'trilinear'] = 'trilinear',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert values into a 3D image with trilinear interpolation.

    Parameters
    ----------
    values: torch.Tensor
        `(...)` array of values to be inserted into `image`.
    coordinates: torch.Tensor
        `(..., 3)` array of 3D coordinates for each value in `data`.
        - Coordinates are ordered `zyx` and are positions in the `d`, `h` and `w` dimensions respectively.
        - Coordinates span the range `[0, N-1]` for a dimension of length N.
    image: torch.Tensor
        `(d, h, w)` array into which data will be inserted.
    weights: torch.Tensor | None
        `(d, h, w)` array containing weights associated with each pixel in `image`.
        This is useful for tracking weights across multiple calls to this function.
    interpolation: Literal['nearest', 'trilinear']
        Interpolation mode used for adding data points to grid.

    Returns
    -------
    image, weights: tuple[torch.Tensor, torch.Tensor]
        The image and weights after updating with data from `data` at `coordinates`.
    """
    if values.shape != coordinates.shape[:-1]:
        raise ValueError('One coordinate triplet is required for each value in data.')
    if coordinates.shape[-1] != 3:
        raise ValueError('Coordinates must be of shape (..., 3).')
    if weights is None:
        weights = torch.zeros_like(image)

    # linearise data and coordinates
    values, _ = einops.pack([values], pattern='*')  # (...) -> (n, )
    coordinates, _ = einops.pack([coordinates], pattern='* zyx')  # (..., 3) -> (n, 3)
    coordinates = coordinates.float()

    # only keep data and coordinates inside the volume
    upper_bound = torch.tensor(image.shape, device=image.device) - 1
    inside = (coordinates >= 0) & (coordinates <= upper_bound)
    inside = torch.all(inside, dim=-1)
    values, coordinates = values[inside], coordinates[inside]

    # splat data onto grid according to interpolation mode
    if interpolation == 'nearest':
        image, weights = _insert_nearest_3d(values, coordinates, image, weights)
    elif interpolation == 'trilinear':
        image, weights = _insert_linear_3d(values, coordinates, image, weights)
    return image, weights


def _insert_nearest_3d(values, coordinates, image, weights):
    coordinates = torch.round(coordinates).long()
    idx_z, idx_y, idx_x = einops.rearrange(coordinates, 'b zyx -> zyx b')
    image.index_put_(indices=(idx_z, idx_y, idx_x), values=values, accumulate=False)
    w = torch.ones(len(coordinates), device=weights.device, dtype=weights.dtype)
    weights.index_put_(indices=(idx_z, idx_y, idx_x), values=w, accumulate=True)
    return image, weights


def _insert_linear_3d(values, coordinates, image, weights):
    # cache floor and ceil of coordinates for each data point being inserted
    _c = torch.empty(size=(values.shape[0], 2, 3), dtype=torch.int64, device=image.device)
    _c[:, 0] = torch.floor(coordinates)  # for lower corners
    _c[:, 1] = torch.ceil(coordinates)  # for upper corners

    # calculate linear interpolation weights for each data point being inserted
    w_dtype = torch.float64 if image.is_complex() else image.dtype
    _w = torch.empty(size=(values.shape[0], 2, 3), dtype=w_dtype, device=image.device
                     )  # (b, 2, zyx)
    _w[:, 1] = coordinates - _c[:, 0]  # upper corner weights
    _w[:, 0] = 1 - _w[:, 1]  # lower corner weights

    # define function for adding weighted data at nearest 8 voxels to each coordinate
    # make sure to do atomic adds, don't just override existing data at each position
    def add_data_at_corner(z: Literal[0, 1], y: Literal[0, 1], x: Literal[0, 1]):
        w = einops.reduce(_w[:, [z, y, x], [0, 1, 2]], 'b zyx -> b', reduction='prod')
        idx_z, idx_y, idx_x = einops.rearrange(_c[:, [z, y, x], [0, 1, 2]], 'b zyx -> zyx b')
        image.index_put_(indices=(idx_z, idx_y, idx_x), values=w * values, accumulate=True)
        weights.index_put_(indices=(idx_z, idx_y, idx_x), values=w, accumulate=True)

    # insert correctly weighted data at each of 8 nearest voxels and return
    add_data_at_corner(0, 0, 0)
    add_data_at_corner(0, 0, 1)
    add_data_at_corner(0, 1, 0)
    add_data_at_corner(0, 1, 1)
    add_data_at_corner(1, 0, 0)
    add_data_at_corner(1, 0, 1)
    add_data_at_corner(1, 1, 0)
    add_data_at_corner(1, 1, 1)
    return image, weights
