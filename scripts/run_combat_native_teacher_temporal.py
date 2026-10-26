"""Expand paired teachers from unused maps and later roots of a sealed collection."""
import argparse
import subprocess
import sys
from pathlib import Path

from process_combat_architecture_pool import read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collection', type=Path, required=True)
    ap.add_argument('--index', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = a.out.resolve()
    assert not out.exists()
    out.mkdir(parents=True)
    collection = a.collection.resolve()
    index = a.index.resolve()
    source = read(collection/'complete.json')
    assert source['state'] == 'complete' and source['collection_only']
    assert not source['training_performed']
    save(out/'protocol.json', dict(
        collection=str(collection), collection_sha256=sha(collection/'complete.json'),
        index=str(index), index_sha256=sha(index), minimum_frame_gap=30,
        max_roots=16, max_instances=16, timescale=2,
        selection='Unused maps first event; reused worlds first eligible event at least 30 frames after last registered root. Fixed SHA map order. No outcome or label filtering.',
        scope='Correlated TRAIN roots from a reused collection, not new independent worlds or quality evaluation.',
        training_performed=False, promotion=False))

    def run(label, script, args):
        save(out/'progress.json', dict(stage=label))
        with (out/(label+'.log')).open('w', encoding='utf-8') as log:
            p = subprocess.run([sys.executable, str(repo/'scripts'/script), *map(str, args)],
                               cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert p.returncode == 0, str(out/(label+'.log'))

    try:
        run('paired', 'run_combat_wide_gate_teacher_pilot.py', [
            '--collection', collection, '--exclude-index', index, '--minimum-frame-gap', 30,
            '--out', out/'paired'])
        paired = read(out/'paired/complete.json')
        assert paired['diagnostic_only'] and not paired['training_performed']
        run('corpus', 'build_combat_native_teacher_registry.py', [
            '--source', out/'paired', '--out', out/'corpus'])
        corpus = read(out/'corpus/complete.json')
        manifest = read(out/'corpus/manifest.json')
        assert corpus['records']*3 == paired['native_cases'] == paired['cuda_cases']
        assert sha(index) == read(out/'protocol.json')['index_sha256']
        save(out/'complete.json', dict(
            state='complete', new_source_native_cases=0, reused_source_cases=source['native_cases'],
            paired_native_cases=paired['native_cases'], paired_cuda_cases=paired['cuda_cases'],
            teacher_records=corpus['records'], counts=manifest['counts'],
            paired_sha256=sha(out/'paired/complete.json'),
            corpus_sha256=sha(out/'corpus/complete.json'),
            manifest_sha256=sha(out/'corpus/manifest.json'),
            training_performed=False, ordinary_ppo_eligible=False, promotion=False))
        save(out/'progress.json', dict(stage='complete'))
        print(out/'complete.json')
    except BaseException as e:
        save(out/'progress.json', dict(stage='failed', error=str(e)))
        raise


if __name__ == '__main__':
    main()
