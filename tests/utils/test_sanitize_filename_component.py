import unittest

from parameterized import param, parameterized

import ballroom_scribe.utils


class TestSanitizeFilenameComponent(unittest.TestCase):
    """Unit tests for the `sanitize_filename_component` function."""

    @parameterized.expand(
        [
            param("safe_name_unchanged", text="large-v3", expected="large-v3"),
            param("ollama_tag", text="qwen2.5:7b-instruct-q6_K", expected="qwen2.5_7b-instruct-q6_K"),
            param("huggingface_id", text="Systran/faster-whisper-large-v3", expected="Systran_faster-whisper-large-v3"),
            param("windows_path_collapses_runs", text=r"C:\models\whisper", expected="C_models_whisper"),
            param("posix_path", text="/home/user/models/whisper", expected="_home_user_models_whisper"),
            param("spaces_and_punctuation", text="my model (v2)!", expected="my_model_v2_"),
        ]
    )
    def test_replaces_unsafe_characters(self, _, text, expected):
        self.assertEqual(ballroom_scribe.utils.sanitize_filename_component(text), expected)

    @parameterized.expand(
        [
            param("empty", text=""),
            param("only_unsafe", text=":/\\"),
        ]
    )
    def test_rejects_text_without_safe_characters(self, _, text):
        with self.assertRaisesRegex(ValueError, "contains no characters that are safe in a file name"):
            ballroom_scribe.utils.sanitize_filename_component(text)


if __name__ == "__main__":
    unittest.main()
