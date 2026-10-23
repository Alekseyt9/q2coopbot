"""Freeze a fresh five-variant endpoint screen after the two PPO updates."""
import argparse
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha
from collect_combat_joint_aim_budget_round import historical_plans
from combat_geometry_ppo import TRAINING_VERSION


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('round_root', 'registry', 'initializer', 'source', 'protocol', 'out'):
        ap.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    ap.add_argument('--port', type=int, default=47700)
    a = ap.parse_args(); assert not a.out.exists()
    repo = Path(__file__).resolve().parents[1]
    protocol = read(a.protocol); assert protocol['version'] == TRAINING_VERSION
    descriptors = [read(a.registry.parent/f) for f in read(a.registry)['files']]
    assert len(descriptors) == 8 and all(e['timescale'] == 2 and 'rules' in e['modes'] for e in descriptors)
    history = historical_plans(repo/'workspace/artifacts')
    used = {s for _, p in history for t in p['tasks'] for s in t.get('seeds', [])}
    offset = 0
    while True:
        assert all(offset < e['splits']['validation']['count'] for e in descriptors)
        seeds = {e['splits']['validation']['start']+offset for e in descriptors}
        if not seeds & used: break
        offset += 1
    assert len(seeds) == 8
    variants = []
    for label, root in (('source', a.source), ('initial', a.initializer), ('B', a.round_root/'B'), ('G', a.round_root/'G')):
        root = root.resolve(); seal = read(root/'complete.json')
        for name, key in (('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')):
            assert sha(root/name) == seal[key]
        r = read(root/'report.json'); assert r['device'] == 'cuda'
        if label in ('B', 'G'):
            assert r['version'] == TRAINING_VERSION and r['round'] == 2 and r['arm'] == label
            assert r['experiment_protocol_sha256'] == sha(a.protocol)
        if label == 'initial': assert seal['weights_sha256'] == protocol['initializer_weights_sha256']
        variants.append(dict(label=label, mode='learned', update=str(root), deterministic=False, cuda_replay=True, checked_wall_slide=True))
    variants.append(dict(label='rules', mode='rules', checked_wall_slide=True))
    comparisons = [(x, y) for x, y in (('source', 'initial'), ('initial', 'B'), ('initial', 'G'), ('B', 'G'), ('rules', 'B'), ('rules', 'G'))]
    spec = dict(version='registered_combat_model_comparison_v1', registry=str(a.registry.resolve()), split='validation', count=1,
        max_instances=16, episodes=sorted(e['id'] for e in descriptors), seed_offset=offset, port=a.port, minimum_free_gib=32,
        ready=[str((a.round_root/'audit/complete.json').resolve())], variants=variants,
        comparisons=[dict(name=x+'-'+y, before=x, after=y) for x, y in comparisons], historical_plans=[str(p) for p, _ in history],
        unavailable_historical_references=historical_plans.missing_references, experiment_protocol=str(a.protocol.resolve()),
        experiment_protocol_sha256=sha(a.protocol), training_performed=False, promotion=False)
    a.out.parent.mkdir(parents=True, exist_ok=True); save(a.out, spec)
    print(dict(seed_offset=offset, native_cases=40, cuda_cases=32, spec=str(a.out)))


if __name__ == '__main__': main()
