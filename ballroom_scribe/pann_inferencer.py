"""PANN sound event detection inference and caching service."""

import collections
import functools
import pathlib
from collections.abc import Iterable
from typing import TYPE_CHECKING

import lazy_loader
import numpy as np

import ballroom_scribe.models
import ballroom_scribe.utils

if TYPE_CHECKING:
    import panns_inference
else:
    # Loaded lazily so `panns_inference/config.py` does not run (and shell out to `wget` for `class_labels_indices.csv`)
    # before `_ensure_panns_data` has downloaded the required files.
    panns_inference = lazy_loader.load("panns_inference")

_PANNS_DATA_DIR_NAME = "panns_data"
_LABELS_CSV_FILENAME = "class_labels_indices.csv"
_LABELS_CSV_URL = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
_CHECKPOINT_FILENAME = "Cnn14_DecisionLevelMax.pth"
_CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"
# `panns_inference.SoundEventDetection.__init__` shells out to `wget` if the checkpoint is smaller than `3e8` bytes.
_MIN_CHECKPOINT_BYTES = 300_000_000

# The PANN is trained on the AudioSet Ontology: https://research.google.com/audioset/ontology/index.html
# Its labels are defined here: https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv
# You can find links to example sounds here: https://github.com/audioset/ontology/blob/master/ontology.json
_CLASSIFICATION_LABEL_INDICES = collections.OrderedDict[ballroom_scribe.models.AudioChunk.Classification, list[int]](
    (
        (ballroom_scribe.models.AudioChunk.Classification.SPEECH, [0, 1, 2, 3, 4, 5]),
        (ballroom_scribe.models.AudioChunk.Classification.MUSIC, list(range(27, 38)) + list(range(137, 283))),
        # [63, 66, 67] covers applause, which might be useful in the future.
    )
)


def _ensure_panns_data() -> pathlib.Path:
    """Downloads `panns_inference`'s label CSV and model checkpoint into `~/panns_data` if missing.

    Returns:
        The path to the downloaded `Cnn14_DecisionLevelMax.pth` model checkpoint.
    """
    panns_data_dir = pathlib.Path.home() / _PANNS_DATA_DIR_NAME
    ballroom_scribe.utils.download_if_missing(_LABELS_CSV_URL, panns_data_dir / _LABELS_CSV_FILENAME)
    checkpoint_path = panns_data_dir / _CHECKPOINT_FILENAME
    ballroom_scribe.utils.download_if_missing(_CHECKPOINT_URL, checkpoint_path, min_bytes=_MIN_CHECKPOINT_BYTES)
    return checkpoint_path


class PannInferencer:
    """Runs the PANN sound event detection model on audio chunks and caches framewise probabilities."""

    SAMPLE_RATE = 32_000
    CLASSIFICATIONS = _CLASSIFICATION_LABEL_INDICES.keys()

    def __init__(self, device: str):
        """Initializes the instance.

        Args:
            device: The PyTorch device on which to run PANN inference, or `"auto"` to use CUDA when it is available
                and the CPU otherwise.
        """
        self._device = ballroom_scribe.utils.resolve_torch_device(device)

    @functools.cached_property
    def _sound_event_detection(self) -> panns_inference.SoundEventDetection:
        """The sound event detection model."""
        checkpoint_path = _ensure_panns_data()
        # Work around a bug in `panns_inference.SoundEventDetection`: its `__init__` checks `device == "cuda"`
        # literally (silently falling back to `"cpu"` for indexed devices such as `"cuda:1"`) and wraps the model in
        # `torch.nn.DataParallel` (which runs single-chunk batches on `cuda:0`). Initialize on `"cpu"` first and move
        # the model onto `self._device` ourselves.
        sed = panns_inference.SoundEventDetection(checkpoint_path=str(checkpoint_path), device="cpu")
        sed.device = self._device
        sed.model.to(self._device)
        return sed

    def infer(
        self,
        audio_chunks: Iterable[ballroom_scribe.models.AudioChunk],
        cache_dir: pathlib.Path,
    ) -> np.typing.NDArray[np.floating]:
        """Computes or loads stitched framewise probabilities across all audio chunks.

        Overlapping regions between consecutive chunks are split evenly at the midpoint, matching the cutover behavior
        of :meth:`ballroom_scribe.AudioClassifier.classify`.

        Args:
            audio_chunks: The audio chunks to process, ordered by start time with no gaps.
            cache_dir: The directory in which inferences are cached.

        Returns:
            A 2D NumPy array of shape `(total_frames, len(PannInferencer.CLASSIFICATIONS))` containing the maximum
            framewise probabilities for each classification category in `PannInferencer.CLASSIFICATIONS`.
        """
        # Storing two classes per frame, at 100 frames per second, consumes 2.88 MB of RAM per hour of audio.
        stitched_segments: list[np.typing.NDArray[np.floating]] = []

        previous_chunk: ballroom_scribe.models.AudioChunk | None = None
        previous_frames: np.typing.NDArray[np.floating] | None = None
        previous_start_frame = 0

        for current_chunk in audio_chunks:
            if not current_chunk:
                continue

            current_frames = self.infer_chunk(
                audio_chunk=current_chunk,
                cache_dir=cache_dir,
            )

            if previous_chunk is not None and previous_frames is not None:
                overlap_cutover = ballroom_scribe.utils.compute_overlap_cutover(
                    previous_chunk.start_samples,
                    previous_chunk.end_samples,
                    current_chunk.start_samples,
                )

                previous_samples_per_frame = previous_chunk.duration_samples / previous_frames.shape[0]
                previous_end_frame = round(
                    (overlap_cutover - previous_chunk.start_samples) / previous_samples_per_frame
                )
                stitched_segments.append(previous_frames[previous_start_frame:previous_end_frame])

                curr_samples_per_frame = current_chunk.duration_samples / current_frames.shape[0]
                previous_start_frame = round((overlap_cutover - current_chunk.start_samples) / curr_samples_per_frame)
            else:
                previous_start_frame = 0

            previous_chunk = current_chunk
            previous_frames = current_frames

        if previous_frames is not None:
            stitched_segments.append(previous_frames[previous_start_frame:])

        if not stitched_segments:
            return np.empty((0, len(self.CLASSIFICATIONS)), dtype=np.float32)

        return np.concatenate(stitched_segments, axis=0)

    def infer_chunk(
        self,
        audio_chunk: ballroom_scribe.models.AudioChunk,
        cache_dir: pathlib.Path,
    ) -> np.typing.NDArray[np.floating]:
        """Loads the per-category max framewise probabilities from the cache, or computes them if not cached.

        Args:
            audio_chunk: The audio chunk on which to run inference.
            cache_dir: The directory in which inferences will be cached.

        Returns:
            The maximum framewise probabilities for each classification category in `PannInferencer.CLASSIFICATIONS`.
            The shape is `(num_frames, len(PannInferencer.CLASSIFICATIONS))`.
        """
        cache_file = cache_dir / f"{audio_chunk.get_filename_stem()}.npy"
        if cache_file.exists():
            return np.load(cache_file)

        audio_data = audio_chunk.data
        # PyTorch doesn't modify the array, but it requires it to be writeable.
        audio_data.flags.writeable = True
        # Although it is possible to run inference on more than one audio clip simultaneously, process only one chunk at
        # a time to reduce VRAM usage.
        framewise_output = self._sound_event_detection.inference(audio_data[None, :])
        audio_data.flags.writeable = False
        # `framewise_output` is a 3D NumPy array. The axes are batches, frames, and labels.
        # `framewise_output.shape[0]` is always 1 because only one chunk is processed at a time.
        # `framewise_output.shape[1]` depends on the length of the audio; longer audio will have more frames.
        # `framewise_output.shape[2]` should always equal 527, as `panns_inference` was trained on that many labels.
        raw_frames = framewise_output[0]  # A 2D view of the sole batch

        # `indices` is a list of `int`s. It represents the indices of the labels that fall under `classification`.
        # For example, the "speech" classification may have labels like "conversation" and "narration."
        # For each classification category, get the highest probability across its label indices for each frame.
        # `frames` has shape `(num_frames, len(_CLASSIFICATION_LABEL_INDICES))`, with columns ordered by
        # `_CLASSIFICATION_LABEL_INDICES`.
        frames = np.column_stack(
            [np.max(raw_frames[:, indices], axis=1) for indices in _CLASSIFICATION_LABEL_INDICES.values()]
        )

        with ballroom_scribe.utils.atomic_path(cache_file) as partial_cache_file:
            np.save(partial_cache_file, frames)
        return frames
