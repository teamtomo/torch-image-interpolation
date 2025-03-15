# torch-image-interpolation

[![License](https://img.shields.io/pypi/l/torch-image-interpolation.svg?color=green)](https://github.com/teamtomo/torch-image-interpolation/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-image-interpolation.svg?color=green)](https://pypi.org/project/torch-image-interpolation)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-image-interpolation.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/torch-image-interpolation/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-image-interpolation/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/torch-image-interpolation/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/torch-image-interpolation)

2D/3D image interpolation and gridding in PyTorch.

## Why?

This package provides a simple, consistent API for

- sampling values from 2D/3D images (`sample_image_2d()`/`sample_image_3d()`)
- inserting values into 2D/3D images (`insert_into_image_2d()`/`insert_into_image_3d`)

Operations are differentiable and sampling from complex valued images is supported.

# Installation

```shell
pip install torch-image-interpolation
```

# Usage

## Sample from image

```python
import torch
import numpy as np
from torch_image_interpolation import sample_image_2d

# example (h, w) image
image = torch.rand((28, 28))

# make an arbitrary stack (..., 2) of 2d coords
coords = torch.tensor(np.random.uniform(low=0, high=27, size=(6, 7, 8, 2))).float()

# sampling returns a (6, 7, 8) array of samples obtained by linear interpolation
samples = sample_image_2d(image=image, coordinates=coords)
```

The API is identical for 3D `(d, h, w)` images but takes `(..., 3)` arrays of
coordinates.

## Insert into image

```python
import torch
import numpy as np
from torch_image_interpolation import insert_into_image_2d

# example (h, w) image
image = torch.zeros((28, 28))

# make an arbitrary stack (..., 2) of 2d coords
coords = torch.tensor(np.random.uniform(low=0, high=27, size=(3, 4, 2)))

# generate random values to place at coords
values = torch.rand(size=(3, 4))

# sampling returns a (6, 7, 8) array of samples obtained by linear interpolation
samples = insert_into_image_2d(values, image=image, coordinates=coords)
```

The API is identical for 3D `(d, h, w)` images but takes `(..., 3)` arrays of
coordinates.

## Coordinate System

This library uses an array-like coordinate system where coordinate values span from `0`
to `dim_size - 1` for each dimension.
Fractional coordinates are supported and values are interpolated appropriately.

### 2D Images

For 2D images with shape `(h, w)`:

Coordinates are ordered as `[y, x]` where:

- `y` is the position in the height dimension (first dimension of shape)
- `x` is the position in the width dimension (second dimension of shape)

For example, in a `(28, 28)` image, valid coordinates range from `[0, 0]` to `[27, 27]`

### 3D Images

For 3D images with shape `(d, h, w)`:

Coordinates are ordered as `[z, y, x]` where:

- `z` is the position in the depth dimension (first dimension of shape)
- `y` is the position in the height dimension (second dimension of shape)
- `x` is the position in the width dimension (third dimension of shape)

For example, in a `(28, 28, 28)` volume, valid coordinates range from `[0, 0, 0]` to
`[27, 27, 27]`.

