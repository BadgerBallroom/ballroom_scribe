import unittest
from unittest.mock import patch

import torch.cuda
import torch.version
from parameterized import param, parameterized

import ballroom_scribe.utils


class TestDescribeTorchDevices(unittest.TestCase):
    """Unit tests for the `describe_torch_devices` function."""

    def setUp(self):
        self.mock_cuda_is_available = self.enterContext(patch.object(torch.cuda, "is_available", autospec=True))
        self.mock_cuda_device_count = self.enterContext(patch.object(torch.cuda, "device_count", autospec=True))
        self.enterContext(patch.object(torch.version, "cuda", "13.2"))

    def test_cuda_unavailable(self):
        self.mock_cuda_is_available.return_value = False

        self.assertEqual(
            ballroom_scribe.utils.describe_torch_devices(),
            "PyTorch: CUDA is not available, so models run on the CPU",
        )

        self.mock_cuda_device_count.assert_not_called()

    @parameterized.expand(
        [
            param("one_device", device_count=1, expected="PyTorch: CUDA 13.2 is available (1 device)"),
            param("two_devices", device_count=2, expected="PyTorch: CUDA 13.2 is available (2 devices)"),
        ]
    )
    def test_cuda_available(self, _, device_count, expected):
        self.mock_cuda_is_available.return_value = True
        self.mock_cuda_device_count.return_value = device_count

        self.assertEqual(ballroom_scribe.utils.describe_torch_devices(), expected)

        self.mock_cuda_is_available.assert_called_once_with()
        self.mock_cuda_device_count.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
