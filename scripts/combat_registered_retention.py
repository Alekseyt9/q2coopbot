"""Verify a train-only registered pool and its closed CUDA processing receipt."""
import pathlib

from process_combat_architecture_pool import read, sha
from process_registered_combat_pool_cuda import select_learned_jobs


VERSION = 'combat_sequence_retention_v2'


def bindings(pool_path, plan_index, processing_path):
    pool_path, processing_path = pathlib.Path(pool_path), pathlib.Path(processing_path)
    pool = read(pool_path)
    assert 0 <= plan_index < len(pool['plans'])
    jobs = select_learned_jobs(pool, plan_index)
    binding = pool['plans'][plan_index]
    plan_path = pathlib.Path(binding['path'])
    assert sha(plan_path) == binding['sha256']
    plan = read(plan_path)
    assert plan['tasks'] and all(t['split'] == 'train' for t in plan['tasks']), 'Train-only histories required'
    model_sha = sha(plan['model_path'])
    assert plan['model_sha256'] == binding['model_sha256'] == model_sha
    expected = {(i, seed, 'learned') for i, task in enumerate(plan['tasks']) for seed in task['seeds']}
    actual = {(j['task_index'], j['seed'], j['mode']) for j in jobs}
    assert actual == expected and len(actual) == len(jobs)
    seed_family = {seed: t['episode']['id'] for t in plan['tasks'] for seed in t['seeds']}
    assert len(seed_family) == len(jobs), 'Seeds must identify unique episodes'
    receipt = read(processing_path)
    assert receipt['state'] == 'complete' and receipt['device'] == 'cuda'
    assert receipt['pool_sha256'] == sha(pool_path) and receipt['model_sha256'] == model_sha
    selected = receipt['selected_plan_index']
    assert selected == plan_index or selected is None and len(pool['plans']) == 1
    cases = receipt['cases']
    assert all(c['split'] == 'train' for c in cases)
    assert {(c['job'], c['seed']) for c in cases} == {(j['job'], j['seed']) for j in jobs}
    assert len(cases) == len(jobs)
    data = processing_path.parent / 'merged'
    meta = read(data / 'report.json')
    assert meta['model_sha256'] == model_sha and meta['rows'] == receipt['rows']
    for name in ('rollout', 'sequence'):
        assert sha(data / (name + '.jsonl')) == meta[name + '_sha256']
    return plan_path, plan, seed_family, data, meta


def validate(spec, meta):
    assert sha(spec['registered_pool']) == spec['pool_sha256']
    assert sha(spec['processing_report']) == spec['processing_report_sha256']
    plan_path, plan, seed_family, _, actual_meta = bindings(
        spec['registered_pool'], spec['plan_index'], spec['processing_report'])
    assert plan_path.resolve() == pathlib.Path(spec['training_plan']).resolve()
    for field in ('model_sha256', 'rollout_sha256', 'sequence_sha256', 'feature_version', 'rows'):
        assert actual_meta[field] == meta[field]
    return seed_family
