"""Regression checks for nested history exclusion and malformed seed records."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess
import os
from types import SimpleNamespace

from combat_plan_history import discover_plan_files, historical_plans, _discover_plan_files_walk


class PlanHistoryTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[1] / 'workspace/build'
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base, prefix='history-test-')
        self.root = Path(self.temp.name).resolve()
        self.addCleanup(self.temp.cleanup)

    def write(self, path, value):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value), encoding='utf-8')
        return target

    def test_deep_previous_run_is_seen_but_current_run_is_excluded(self):
        old = self.write('previous/round-04/observed/corpus/train-plan.json', {'tasks': [{'seeds': [16]}]})
        self.write('current/evaluation/plan.json', {'tasks': [{'seeds': [17]}]})
        self.write('plan-list.json', ['metadata-only'])
        records = list(historical_plans(self.root, [old], [self.root / 'current']))
        self.assertEqual([(p, d['tasks'][0]['seeds']) for p, d in records], [(old, [16])])

    def test_explicit_external_history_need_not_match_filename_pattern(self):
        external = self.write('outside/receipt.json', {'tasks': [{'seeds': [23]}]})
        self.assertEqual([p for p, _ in historical_plans(self.root / 'missing', [external])], [external])

    def test_malformed_or_missing_history_is_not_silently_ignored(self):
        path = self.write('plan.json', {'tasks': [{'seeds': ['23']}]})
        with self.assertRaisesRegex(ValueError, 'Invalid historical seed'):
            list(historical_plans(self.root))
        with self.assertRaises(FileNotFoundError):
            list(historical_plans(self.root / 'missing', [self.root / 'absent.json']))
        path.write_text('{', encoding='utf-8')
        with self.assertRaises(json.JSONDecodeError):
            list(historical_plans(self.root))

    def test_directory_alias_is_not_followed(self):
        target = self.root / 'outside'
        self.write('outside/plan.json', {'tasks': [{'seeds': [29]}]})
        scan = self.root / 'scan'
        scan.mkdir()
        try:
            (scan / 'alias').symlink_to(target, target_is_directory=True)
        except OSError:
            self.skipTest('Directory symlink privilege unavailable; junction checked separately on Windows')
        self.assertEqual(discover_plan_files(scan), [])

    def test_native_and_fallback_discovery_match(self):
        self.write('nested space/План/UPPER-PLAN.JSON', {'tasks': [{'seeds': [31]}]})
        self.write('.hidden/x-plan.json', {'tasks': [{'seeds': [32]}]})
        self.write('excluded/plan.json', {'tasks': [{'seeds': [33]}]})
        excluded=[self.root/'excluded']
        expected=_discover_plan_files_walk(self.root,excluded)
        self.assertEqual(discover_plan_files(self.root,excluded),expected)
        with patch('combat_plan_history.shutil.which',return_value=None):
            self.assertEqual(discover_plan_files(self.root,excluded),expected)

    @unittest.skipUnless(os.name=='nt','Windows native discovery')
    def test_failed_native_search_is_not_empty_history(self):
        with patch('combat_plan_history.shutil.which',return_value='rg'), patch(
                'combat_plan_history.subprocess.run',return_value=SimpleNamespace(returncode=2,stdout='',stderr='Access denied')):
            with self.assertRaisesRegex(RuntimeError,'History file discovery failed'):
                discover_plan_files(self.root)

    @unittest.skipUnless(os.name=='nt','Windows junction contract')
    def test_windows_junction_is_not_followed(self):
        self.write('outside/plan.json', {'tasks': [{'seeds': [34]}]})
        scan=self.root/'scan';scan.mkdir();alias=scan/'junction'
        quote=lambda p:"'"+str(p).replace("'","''")+"'"
        subprocess.run(['pwsh','-NoProfile','-Command',
            'New-Item -ItemType Junction -Path '+quote(alias)+' -Target '+quote(self.root/'outside')+' | Out-Null'],
            check=True,capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
        self.addCleanup(lambda: os.rmdir(alias) if alias.exists() else None)
        self.assertEqual(discover_plan_files(scan),[])


if __name__ == '__main__':
    unittest.main()
