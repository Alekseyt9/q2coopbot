import pathlib
import tempfile
import unittest
from compare_registered_combat_pools import runtime_bindings,sha


class RuntimeBindingsTests(unittest.TestCase):
    def test_requires_native_module_and_detects_changed_file(self):
        cache=pathlib.Path(__file__).resolve().parents[1]/'workspace/build/training-cache'
        cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            root=pathlib.Path(directory);runtime=root/'runtime';(runtime/'baseq2').mkdir(parents=True)
            for name in ('q2ded.exe','baseq2/game.dll'):(runtime/name).write_bytes(name.encode())
            entries=[dict(path=name,sha256=sha(runtime/name)) for name in ('q2ded.exe','baseq2/game.dll')]
            row=dict(root=str(root),runtime_files=entries)
            self.assertEqual(set(runtime_bindings(row)),{'q2ded.exe','baseq2/game.dll'})
            with self.assertRaises(AssertionError):runtime_bindings({**row,'runtime_files':entries[:1]})
            with self.assertRaises(AssertionError):runtime_bindings({**row,'runtime_files':entries+[entries[0]]})
            (runtime/'baseq2/game.dll').write_bytes(b'other native build')
            with self.assertRaises(AssertionError):runtime_bindings(row)


if __name__=='__main__':unittest.main()
