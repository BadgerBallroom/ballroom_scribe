"""Utility for releasing GPU memory between VRAM-heavy pipeline stages."""

import gc

import torch
import torch.cuda


def free_vram() -> None:
    """Runs garbage collection and releases cached CUDA memory back to the device.

    Call this after the last reference to a VRAM-consuming model (e.g. `AudioClassifier` or `SpeechTranscriber`) has
    been dropped so that the next stage can load its own model. The CUDA cache step is skipped when CUDA is unavailable.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
