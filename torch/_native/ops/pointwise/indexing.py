# Admission checks for the signed 32-bit pointwise kernels.

from __future__ import annotations

import math
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


def can_use_32bit_index_math(
    tensors: Sequence[torch.Tensor], shape: Sequence[int], tile: int = 8192
) -> bool:
    # Include inactive tail lanes; 8192 bounds the default kernel launch tiles.
    limit = (1 << 31) - 1
    numel = math.prod(shape)
    if ((numel + tile - 1) // tile) * tile > limit:
        return False
    return all(
        sum((size - 1) * abs(stride) for size, stride in zip(t.shape, t.stride()))
        <= limit
        for t in tensors
    )
