"""Split and lossless compressed evidence checks; no neural execution."""
import gzip
import tempfile
import unittest
from pathlib import Path
from prepare_combat_shot_time_corpus import diagnostic_worlds, compressed_copy


class Corpus(unittest.TestCase):
    def test_paired_models_share_world_split(self):
        variants = []
        for name, base in [('base3', 1000), ('city1', 2000)]:
            for site in range(2):
                for offset in range(2):
                    for model in range(2):
                        variants.append(dict(context=dict(map=name, seed=base+site*10+offset, fresh_offset=offset), model=model))
        self.assertEqual(diagnostic_worlds(variants), {('base3', 1001), ('city1', 2001)})

    def test_compressed_evidence_is_lossless(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'workspace/artifacts') as folder:
            root = Path(folder)
            data = b'{"observed":1}\n{"observed":2}\n'
            source = root/'source.jsonl'; source.write_bytes(data)
            compressed_copy(source, root/'a.gz'); compressed_copy(source, root/'b.gz')
            self.assertEqual(gzip.decompress((root/'a.gz').read_bytes()), data)
            self.assertEqual((root/'a.gz').read_bytes(), (root/'b.gz').read_bytes())


if __name__ == '__main__':
    unittest.main()
