"""Verify fresh rules provenance, export canonical Go features, gzip new data."""
import argparse
import gzip
import json
import os
import pathlib
import subprocess

from process_combat_architecture_pool import read, save, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    root, out = args.corpus.resolve(), args.out.resolve()
    assert out.is_relative_to(repo / 'workspace/artifacts') and not out.exists()
    pins = {}
    def pin(path, expected=None):
        path = pathlib.Path(path).resolve(); digest = sha(path)
        assert expected is None or expected == digest
        pins[str(path)] = digest
        return path
    seal = read(pin(root / 'complete.json'))
    assert seal['state'] == 'complete' and seal['version'] == 'registered_rules_teacher_corpus_v1'
    protocol = read(pin(root / 'protocol.json', seal['protocol_sha256']))
    assert protocol['test_deferred'] and not seal['neural_execution'] and not seal['training_performed']
    pin(root / 'pool/report.json', seal['pool_sha256'])
    for item in seal['inputs'] + protocol['inputs']:
        pin(item['path'], item['sha256'])
    for path in sorted((repo / 'internal').rglob('*.go')):
        pin(path)
    pin(repo / 'cmd/q2joint-teacher-data/main.go')
    out.mkdir(parents=True)
    request = out / 'request.json'
    save(request, dict(version='registered_joint_teacher_export_request_v1', cases=[
        {k: c[k] for k in ('root', 'seed', 'split', 'network_release_frame')} for c in seal['cases']]))
    pin(request)
    os.environ['GOCACHE'] = str(repo / 'workspace/build/go-cache')
    os.environ['GOTOOLCHAIN'] = 'auto'
    def run(label, command):
        with (out / (label + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(list(map(str, command)), cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert result.returncode == 0, str(out / (label + '.log'))
    binary = out / 'q2joint-teacher-data.exe'
    run('build', ['go', 'build', '-o', binary, './cmd/q2joint-teacher-data'])
    native = out / 'native'
    save(out / 'progress.json', dict(stage='native-proof-and-features', training_performed=False))
    run('export', [binary, '--request', request, '--out', native])
    report = read(pin(native / 'report.json'))
    assert report['state'] == 'complete' and report['cases'] == len(seal['cases'])
    splits = {}
    for split in ('train', 'development'):
        source = native / (split + '.jsonl'); digest = sha(source)
        target = native / (split + '.jsonl.gz')
        payload = source.read_bytes()
        with gzip.open(target, 'wb', compresslevel=6) as stream:
            stream.write(payload)
        with gzip.open(target, 'rb') as stream:
            assert stream.read() == payload
        rows = [json.loads(line) for line in payload.splitlines()]
        assert len(rows) == report['splits'][split]['rows']
        splits[split] = dict(path=str(target), sha256=sha(target), uncompressed_sha256=digest,
            rows=len(rows), bytes=target.stat().st_size, counts=report['splits'][split])
        # Only this fresh, verified, derived output; raw captures are untouched.
        source.unlink()
    for path, digest in pins.items():
        assert sha(path) == digest
    save(out / 'complete.json', dict(state='complete', version='registered_joint_teacher_features_v1',
        corpus=str(root), cases=len(seal['cases']), feature_version='combat_features_v10',
        feature_width=1147, test_deferred=True, splits=splits,
        inputs=[dict(path=p, sha256=h) for p, h in pins.items()], script_sha256=sha(__file__),
        training_performed=False, neural_execution=False, movement_supervision=False,
        vertical_supervision=False, promotion=False))
    save(out / 'progress.json', dict(stage='complete', training_performed=False))
    print(json.dumps(dict(state='complete', cases=len(seal['cases']), splits=splits)))


if __name__ == '__main__':
    main()
