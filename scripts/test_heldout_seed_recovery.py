"""Seed provenance recovery gates; no model execution."""
import json,pathlib,tempfile,unittest
from audit_combat_heldout_seeds import recover_interrupted_metadata
from process_combat_architecture_pool import sha

class RecoveryTest(unittest.TestCase):
    def setUp(self):
        scratch=pathlib.Path(__file__).resolve().parents[1]/'workspace/build'
        scratch.mkdir(exist_ok=True,parents=True)
        self.temp=tempfile.TemporaryDirectory(dir=scratch)
        self.repo=pathlib.Path(self.temp.name)
        base=self.repo/'workspace/artifacts'
        self.native=base/'apool-v1-20261007/m6/case-19-learned/s-123'
        self.native.mkdir(parents=True)
        def save(path,obj):path.write_text(json.dumps(obj),encoding='utf-8')
        self.save=save
        save(self.native/'report.json',dict(capture_complete=True,provenance_valid=True,results=[dict(seed=123)]))
        save(self.native/'manifest.json',dict(seeds=[123]))
        member=dict(root=str(self.native),seed=123,report_sha256=sha(self.native/'report.json'),manifest_sha256=sha(self.native/'manifest.json'))
        archived=base/'aproc-v1-20261007/interrupted-m6-123'
        case=archived/'case-19';case.mkdir(parents=True)
        save(case/'manifest.json',dict(seeds=[123],pool_members=[member]))
        save(case/'report.json',dict(provenance_valid=True,capture_complete=True,results=[dict(seed=123)],pool_members=[member]))
        self.rollout=archived/'rollout-19';self.rollout.mkdir()
        self.damaged=self.rollout/'report.json';self.damaged.write_bytes(b'\0'*128)
        (self.rollout/'sequence.jsonl').write_text('{"seed":123}\n',encoding='utf-8')
    def tearDown(self):self.temp.cleanup()
    def test_complete_receipts_bound_zero_report(self):
        result=recover_interrupted_metadata(self.damaged,self.repo,{})
        self.assertEqual(result['seeds'],[123])
        self.assertEqual(result['sequence_rows'],1)
        self.assertEqual(self.damaged.read_bytes(),b'\0'*128)
    def test_unrecorded_sequence_seed_rejected(self):
        (self.rollout/'sequence.jsonl').write_text('{"seed":124}\n',encoding='utf-8')
        with self.assertRaises(AssertionError):recover_interrupted_metadata(self.damaged,self.repo,{})
    def test_changed_native_receipt_rejected(self):
        self.save(self.native/'report.json',dict(capture_complete=True,provenance_valid=True,results=[dict(seed=124)]))
        with self.assertRaises(AssertionError):recover_interrupted_metadata(self.damaged,self.repo,{})
    def test_nonzero_damage_not_waived(self):
        self.damaged.write_bytes(b'broken JSON')
        with self.assertRaises(AssertionError):recover_interrupted_metadata(self.damaged,self.repo,{})

if __name__=='__main__':unittest.main()
