# torch-image-lerp

[![License](https://img.shields.io/pypi/l/torch-image-lerp.svg?color=green)](https://github.com/teamtomo/torch-image-lerp/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-image-lerp.svg?color=green)](https://pypi.org/project/torch-image-lerp)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-image-lerp.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/torch-image-lerp/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-image-lerp/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/torch-image-lerp/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/torch-image-lerp)

Linear 2D/3D image interpolation and gridding in PyTorch.

## Why?

This package provides a simple, consistent API for 
- sampling from 2D/3D images (`sample_image_2d()`/`sample_image_3d()`)
- inserting values into 2D/3D images (`insert_into_image_2d()`, `insert_into_image_3d`)

Operations are differentiable and sampling from complex valued images is supported.

# Installation

```shell
pip install torch-image-lerp
```

# Usage

## Sample from image

```python
import torch
import numpy as np
from torch_image_lerp import sample_image_2d

image = torch.rand((28, 28))

# make an arbitrary stack (..., 2) of 2d coords
coords = torch.tensor(np.random.uniform(low=0, high=27, size=(6, 7, 8, 2))).float()

# sampling returns a (6, 7, 8) array of samples obtained by linear interpolation
samples = sample_image_2d(image=image, coordinates=coords)
```

The API is identical for 3D but takes `(..., 3)` coordinates and a `(d, h, w)` image.

## Insert into image

```python
import torch
import numpy as np
from torch_image_lerp import insert_into_image_2d

image = torch.zeros((28, 28))

# make an arbitrary stack (..., 2) of 2d coords
coords = torch.tensor(np.random.uniform(low=0, high=27, size=(3, 4, 2)))

# generate random values to place at coords
values = torch.rand(size=(3, 4))

# sampling returns a (6, 7, 8) array of samples obtained by linear interpolation
samples = insert_into_image_2d(values, image=image, coordinates=coords)
```

The API is identical for 3D but takes `(..., 3)` coordinates and a `(d, h, w)` image.
