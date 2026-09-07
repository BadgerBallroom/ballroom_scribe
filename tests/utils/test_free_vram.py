import gc
import unittest
from unittest.mock import patch

import torch.cuda

import ballroom_scribe.utils


class TestFreeVram(unittest.TestCase):
    """Unit tests for the `free_vram` function."""

    def setUp(self):
        self.mock_gc_collect = self.enterContext(patch.object(gc, "collect", autospec=True))
        self.mock_cuda_is_available = self.enterContext(patch.object(torch.cuda, "is_available", autospec=True))
        self.mock_cuda_empty_cache = self.enterContext(patch.object(torch.cuda, "empty_cache", autospec=True))

    def test_cuda_available(self):
        self.mock_cuda_is_available.return_value = True

        ballroom_scribe.utils.free_vram()

        self.mock_gc_collect.assert_called_once_with()
        self.mock_cuda_empty_cache.assert_called_once_with()

    def test_cuda_unavailable(self):
        self.mock_cuda_is_available.return_value = False

        ballroom_scribe.utils.free_vram()

        self.mock_gc_collect.assert_called_once_with()
        self.mock_cuda_empty_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
