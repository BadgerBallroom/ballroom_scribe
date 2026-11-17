"""Utility for choosing the PyTorch device on which a model runs."""

import torch
import torch.cuda

import ballroom_scribe.utils

_AUTO_DEVICE = "auto"


def resolve_torch_device(device: str) -> str:
    """Resolves `"auto"` to a concrete PyTorch device and validates explicit devices.

    Args:
        device: `"auto"` to use `"cuda"` when CUDA is available and `"cpu"` otherwise, or an explicit PyTorch device
            string such as `"cpu"`, `"cuda"`, or `"cuda:1"`.

    Returns:
        The concrete device string. `"auto"` resolves to plain `"cuda"` rather than `"cuda:0"` because
        `panns_inference.SoundEventDetection` only uses the GPU when its device is exactly `"cuda"`.

    Raises:
        ValueError: If an explicit device requests CUDA while CUDA is not available, or names a CUDA device index that
            does not exist.
    """
    if device == _AUTO_DEVICE:
        return "cuda" if torch.cuda.is_available() else "cpu"
    ballroom_scribe.utils.validate_torch_device(device)
    return device
