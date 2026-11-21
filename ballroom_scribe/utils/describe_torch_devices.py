"""Utility that describes the devices that PyTorch can run models on, for logs and problem reports."""

import torch
import torch.cuda
import torch.version


def describe_torch_devices() -> str:
    """Returns one line saying whether PyTorch can use CUDA and how many CUDA devices it sees.

    This asks only what `resolve_torch_device` asks as well. No CUDA context is created, so no VRAM is taken.
    """
    if not torch.cuda.is_available():
        return "PyTorch: CUDA is not available, so models run on the CPU"
    device_count = torch.cuda.device_count()
    devices = "1 device" if device_count == 1 else f"{device_count} devices"
    return f"PyTorch: CUDA {torch.version.cuda} is available ({devices})"
