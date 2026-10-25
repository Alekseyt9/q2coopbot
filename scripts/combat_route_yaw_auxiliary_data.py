"""Explicit v2 data admission for geometric auxiliaries, never PPO readiness."""
import json
from pathlib import Path
from train_combat_memory_route_ppo import admit as admit_v1
from evaluate_registered_combat_models import read, sha


def admit(directory, model_hash, consumed):
    meta = read(directory/'report.json')
    if meta.get('memory_route_movement_version') == 'combat_memory_route_movement_v1':
        return admit_v1(directory, model_hash, consumed)
    proof = read(directory/'cuda-verification.json')
    assert meta.get('memory_route_movement_version') == 'combat_memory_route_movement_yaw_v2'
    assert meta['version'] == 'combat_ppo_rollout_v1' and meta['numerical_verification'] == 'cuda_verified_v1'
    assert meta['feature_version'] == 'combat_features_v13_memory_static_routes' and meta['model_sha256'] == model_hash
    assert meta['reward_version'] == 'combat_reward_v14' and meta['observation_timing_contract'] == 'frozen_world_v1'
    assert not meta.get('diagnostic_only') and not any(k.startswith('diagnostic_') for k in meta)
    assert proof['state'] == 'passed' and proof['device'] == 'cuda' and proof['model_sha256'] == model_hash
    assert sha(directory/'cuda-verification.json') == meta['cuda_verification_sha256']
    assert proof['execution_contract_verified'] and proof['ordinary_training_eligible'] is False
    assert meta['memory_route_training_supported'] is False and meta['ppo_training_ready'] is False
    for name in ('rollout', 'sequence'):
        assert sha(directory/f'{name}.jsonl') == meta[f'{name}_sha256']
    token = meta['native_rollout_sha256']
    assert token not in consumed, 'Rollout already consumed'
    for path, digest in meta['source_sha256'].items():
        assert sha(Path(path)) == digest
    def load(name):
        with (directory/f'{name}.jsonl').open(encoding='utf-8-sig') as f:
            return [json.loads(line) for line in f]
    rows, history = load('rollout'), load('sequence')
    assert len(rows) == meta['rows'] > 0 and len(history) == meta['sequence_rows']
    assert all(len(r['features']) == 1554 and r['sample']['version'] == meta['policy_version'] for r in rows)
    return rows, history, token, meta
