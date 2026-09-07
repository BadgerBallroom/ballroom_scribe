import torch
import torch.cuda


def validate_torch_device(device: str) -> None:
    """Validate the torch device. Raises a ValueError if the device is invalid."""
    d = torch.device(device=device)

    if d.type != "cuda":
        return

    if not torch.cuda.is_available():
        raise ValueError("The device is set to CUDA, but CUDA is not available.")

    if d.index is not None and d.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {d.index} is out of range (found {torch.cuda.device_count()} device(s)).")
