import json
import pathlib
import tempfile
import unittest
from train_combat_hjepa_branch_preferences import load_native


class NativeCalibrationDataTests(unittest.TestCase):
    def test_confirmation_cannot_be_loaded_as_training(self):
        with tempfile.TemporaryDirectory() as root:
            p = pathlib.Path(root) / 'data.jsonl'
            pathlib.Path(str(p) + '.meta.json').write_text(json.dumps(dict(
                version='hjepa_native_weapon_preference_data_v1', split='confirmation', diagnostic=True)))
            with self.assertRaisesRegex(ValueError, 'contract/split'):
                load_native(p, 'train')

    def test_modified_native_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            p = pathlib.Path(root) / 'data.jsonl'
            receipt = pathlib.Path(root) / 'report.json'
            receipt.write_text('{}')
            pathlib.Path(str(p) + '.meta.json').write_text(json.dumps(dict(
                version='hjepa_native_weapon_preference_data_v1', split='train', diagnostic=True,
                sources_sha256={}, native_receipts_sha256={str(receipt): 'incorrect'})))
            with self.assertRaisesRegex(ValueError, 'source/receipt'):
                load_native(p, 'train')


if __name__ == '__main__':
    unittest.main()
