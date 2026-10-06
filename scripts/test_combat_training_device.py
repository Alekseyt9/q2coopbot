import unittest
from unittest.mock import patch
from train_combat_bc import training_devices


class TrainingDeviceTests(unittest.TestCase):
    def test_available_cuda_is_only_training_device(self):
        with patch('train_combat_bc.torch.cuda.is_available',return_value=True):
            self.assertEqual(training_devices(),['cuda'])

    def test_missing_cuda_stops_training(self):
        with patch('train_combat_bc.torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CPU training is disabled'):
                training_devices()


if __name__ == '__main__': unittest.main()
