import unittest
from unittest.mock import patch

import torch.cuda
from parameterized import param, parameterized

import ballroom_scribe.utils


class TestResolveTorchDevice(unittest.TestCase):
    """Unit tests for the `resolve_torch_device` function."""

    def setUp(self):
        self.mock_torch_cuda_is_available = self.enterContext(patch.object(torch.cuda, "is_available", autospec=True))
        self.mock_validate_torch_device = self.enterContext(
            patch.object(ballroom_scribe.utils, "validate_torch_device", autospec=True)
        )

    @parameterized.expand(
        [
            param("cuda_available", cuda_available=True, expected="cuda"),
            param("cuda_unavailable", cuda_available=False, expected="cpu"),
        ]
    )
    def test_auto_follows_cuda_availability(self, _, cuda_available, expected):
        self.mock_torch_cuda_is_available.return_value = cuda_available

        self.assertEqual(ballroom_scribe.utils.resolve_torch_device("auto"), expected)

        self.mock_torch_cuda_is_available.assert_called_once_with()
        self.mock_validate_torch_device.assert_not_called()

    @parameterized.expand(
        [
            param("cpu", device="cpu"),
            param("cuda", device="cuda"),
            param("cuda_index", device="cuda:1"),
        ]
    )
    def test_explicit_device_is_validated_and_returned_unchanged(self, _, device):
        self.assertEqual(ballroom_scribe.utils.resolve_torch_device(device), device)

        self.mock_validate_torch_device.assert_called_once_with(device)
        self.mock_torch_cuda_is_available.assert_not_called()

    def test_explicit_device_validation_error_propagates(self):
        self.mock_validate_torch_device.side_effect = ValueError(
            "The device is set to CUDA, but CUDA is not available."
        )

        with self.assertRaisesRegex(ValueError, "CUDA is not available"):
            ballroom_scribe.utils.resolve_torch_device("cuda")


if __name__ == "__main__":
    unittest.main()
