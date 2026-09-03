"""Bit-packing for binary grid observations.

Team/1v1 foraging observations are (H, W, F) int8 with strictly binary
channels (F <= 8). On the large MovingAI maps a raw state is ~37KB (81x65x7),
which makes offline datasets huge; np.packbits squeezes F channels into one
uint8 per cell (8x smaller, lossless).

pack_obs/unpack_obs are exact inverses for binary inputs:
    unpack_obs(pack_obs(obs), F) == obs
"""

import numpy as np


def pack_obs(obs: np.ndarray) -> np.ndarray:
    """(H, W, F) binary int8/uint8 obs -> (H, W, 1) uint8 bit-packed."""
    assert obs.ndim == 3 and obs.shape[2] <= 8, (
        f"pack_obs expects (H, W, F<=8), got {obs.shape}"
    )
    return np.packbits(obs.astype(bool), axis=2)


def unpack_obs(packed: np.ndarray, features: int = 7) -> np.ndarray:
    """(..., H, W, 1) uint8 -> (..., H, W, features) uint8."""
    assert packed.shape[-1] == 1, (
        f"unpack_obs expects (..., H, W, 1), got {packed.shape}"
    )
    return np.unpackbits(packed, axis=-1, count=features)


def pack_obs_batch(obs_list) -> np.ndarray:
    """List of (H, W, F) binary obs -> (B, H, W, 1) uint8 stacked + packed."""
    return np.stack([pack_obs(o) for o in obs_list])
