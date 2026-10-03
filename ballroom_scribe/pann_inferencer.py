"""PANN sound event detection inference and caching service."""

import collections
import functools
import pathlib
from collections.abc import Iterable

import numpy as np
import panns_inference

import ballroom_scribe.models
import ballroom_scribe.utils

# The PANN is trained on the AudioSet Ontology: https://research.google.com/audioset/ontology/index.html
# Its labels are defined here: http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv
# You can find links to example sounds here: https://github.com/audioset/ontology/blob/master/ontology.json
_CLASSIFICATION_LABEL_INDICES = collections.OrderedDict[ballroom_scribe.models.AudioChunk.Classification, list[int]](
    (
        (ballroom_scribe.models.AudioChunk.Classification.SPEECH, [0, 1, 2, 3, 4, 5]),
        (ballroom_scribe.models.AudioChunk.Classification.MUSIC, list(range(27, 38)) + list(range(137, 283))),
        # [63, 66, 67] covers applause, which might be useful in the future.
    )
)


class PannInferencer:
    """Runs the PANN sound event detection model on audio chunks and caches framewise probabilities."""

    CLASSIFICATIONS = _CLASSIFICATION_LABEL_INDICES.keys()

    def __init__(self, device: str):
        """Initializes the instance.

        Args:
            device: The PyTorch device on which to run PANN inference.
        """
        ballroom_scribe.utils.validate_torch_device(device)
        self._device = device

    @functools.cached_property
    def _sound_event_detection(self) -> panns_inference.SoundEventDetection:
        """The sound event detection model."""
        return panns_inference.SoundEventDetection(device=self._device)

    def infer(
        self,
        audio_chunks: Iterable[ballroom_scribe.models.AudioChunk],
        channel: int,
        cache_dir: pathlib.Path,
    ) -> np.typing.NDArray[np.floating]:
        """Computes or loads stitched framewise probabilities across all audio chunks.

        Overlapping regions between consecutive chunks are split evenly at the midpoint, matching the cutover behavior
        of :meth:`ballroom_scribe.AudioClassifier.classify`.

        Args:
            audio_chunks: The audio chunks to process, ordered by start time with no gaps.
            channel: The audio channel from which `audio_chunks` were loaded.
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
                channel=channel,
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
        channel: int,
        cache_dir: pathlib.Path,
    ) -> np.typing.NDArray[np.floating]:
        """Loads the per-category max framewise probabilities from the cache, or computes them if not cached.

        Args:
            audio_chunk: The audio chunk on which to run inference.
            channel: The audio channel from which `audio_chunk` was loaded.
            cache_dir: The directory in which inferences will be cached.

        Returns:
            The maximum framewise probabilities for each classification category in `PannInferencer.CLASSIFICATIONS`.
            The shape is `(num_frames, len(PannInferencer.CLASSIFICATIONS))`.
        """
        cache_file = cache_dir / f"{audio_chunk.get_filename_stem()}-{channel}.npy"
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
