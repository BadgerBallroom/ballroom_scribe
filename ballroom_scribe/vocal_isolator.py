"""Utilities for isolating human vocals from other audio."""

import functools
import pathlib

import demucs.api
import demucs.audio

import ballroom_scribe.utils


class VocalIsolator:
    """Isolates vocal tracks from audio files using Demucs."""

    def __init__(self, demucs_name: str, demucs_device: str):
        """Initializes the instance.

        Args:
            demucs_name: The name of the Demucs model to use.
            demucs_device: The device to use for Demucs.
        """
        # Validate the arguments now because Demucs is lazily loaded.
        available_models = demucs.api.list_models()
        if demucs_name not in available_models["bag"] and demucs_name not in available_models["single"]:
            raise ValueError(f"Unknown Demucs model: {demucs_name!r}.")

        ballroom_scribe.utils.validate_torch_device(demucs_device)

        self._demucs_name = demucs_name
        self._demucs_device = demucs_device

    @functools.cached_property
    def _separator(self) -> demucs.api.Separator:
        """Initializes the Demucs separator."""
        return demucs.api.Separator(model=self._demucs_name, device=self._demucs_device)

    def separate(self, input_audio: pathlib.Path, output_vocals: pathlib.Path) -> None:
        """Separates vocal tracks using Demucs if the vocal file doesn't already exist."""
        if output_vocals.exists():
            print("Vocals already isolated. Not isolating again.")
        else:
            self._call_demucs(input_audio, output_vocals)

    def _call_demucs(self, input_audio: pathlib.Path, output_vocals: pathlib.Path) -> None:
        _, separated = self._separator.separate_audio_file(input_audio)

        try:
            vocals = separated["vocals"]
        except KeyError as e:
            raise ValueError("The audio did not contain vocals.") from e
        else:
            with ballroom_scribe.utils.atomic_path(output_vocals) as partial_vocals:
                demucs.audio.save_audio(vocals, partial_vocals, samplerate=self._separator.samplerate)
