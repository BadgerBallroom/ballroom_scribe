import unittest
from unittest.mock import patch

import torch.cuda

import ballroom_scribe.utils


class TestValidateTorchDevice(unittest.TestCase):
    """Unit tests for the `validate_torch_device` function."""

    def setUp(self):
        self.mock_torch_cuda_is_available = self.enterContext(patch.object(torch.cuda, "is_available", autospec=True))
        self.mock_torch_cuda_is_available.return_value = True

        self.mock_torch_cuda_device_count = self.enterContext(patch.object(torch.cuda, "device_count", autospec=True))
        self.mock_torch_cuda_device_count.return_value = 1

    def test_does_not_validate_cuda_if_device_is_cpu(self):
        ballroom_scribe.utils.validate_torch_device("cpu")

        self.mock_torch_cuda_is_available.assert_not_called()
        self.mock_torch_cuda_device_count.assert_not_called()

    def test_validates_cuda_available(self):
        self.mock_torch_cuda_is_available.return_value = False

        with self.assertRaisesRegex(ValueError, "The device is set to CUDA, but CUDA is not available."):
            ballroom_scribe.utils.validate_torch_device("cuda")

        self.mock_torch_cuda_is_available.assert_called()

    def test_validates_cuda_device_index(self):
        with self.assertRaisesRegex(ValueError, "CUDA device index 1 is out of range"):
            ballroom_scribe.utils.validate_torch_device("cuda:1")

        self.mock_torch_cuda_device_count.assert_called()


if __name__ == "__main__":
    unittest.main()
