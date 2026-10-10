import json
import pathlib
import tempfile
import unittest
from unittest import mock

from process_combat_architecture_pool import save


class AtomicProgressTests(unittest.TestCase):
    def test_transient_windows_reader_lock_preserves_old_json(self):
        root=pathlib.Path(__file__).resolve().parents[1]/'workspace/build/training-cache'
        root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            path=pathlib.Path(temporary)/'progress.json';save(path,{'old':True})
            replace=pathlib.Path.replace
            attempts=[]
            def locked(pending,target):
                attempts.append(1)
                if len(attempts)<=2:
                    self.assertEqual(json.loads(path.read_text()),{'old':True})
                    error=PermissionError('reader lock');error.winerror=5;raise error
                return replace(pending,target)
            with mock.patch.object(pathlib.Path,'replace',locked),mock.patch('process_combat_architecture_pool.time.sleep') as sleep:
                save(path,{'new':True})
                self.assertEqual(sleep.call_count,2)
            self.assertEqual(json.loads(path.read_text()),{'new':True})
            self.assertFalse(path.with_name('progress.json.partial').exists())

    def test_persistent_error_is_bounded_and_keeps_old_file(self):
        root=pathlib.Path(__file__).resolve().parents[1]/'workspace/build/training-cache'
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            path=pathlib.Path(temporary)/'progress.json';save(path,{'old':True})
            error=PermissionError('lock');error.winerror=32
            with mock.patch.object(pathlib.Path,'replace',side_effect=error) as replace,mock.patch('process_combat_architecture_pool.time.sleep'):
                with self.assertRaises(PermissionError):save(path,{'new':True})
                self.assertEqual(replace.call_count,51)
            self.assertEqual(json.loads(path.read_text()),{'old':True})
            self.assertTrue(path.with_name('progress.json.partial').exists())


if __name__=='__main__':unittest.main()
