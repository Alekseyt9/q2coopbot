"""Audit a full-actor resumed update on CUDA; gameplay quality is separate."""
import argparse
import math
import pathlib

from ppo_recurrent import torch, training_devices, read, sha, durable_json


def main():
    parser = argparse.ArgumentParser()
    for name in ('before', 'after', 'report', 'spec', 'out'):
        parser.add_argument('--' + name, type=pathlib.Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    training_devices()
    before = torch.load(args.before, map_location='cuda', weights_only=True)
    after = torch.load(args.after, map_location='cuda', weights_only=True)
    report, spec = read(args.report), read(args.spec)
    assert report['device'] == 'cuda' and report['actor_training']['scope'] == 'all'
    assert before['version'] == after['version'] == 'combat_architecture_checkpoint_v1'
    assert before['architecture'] == after['architecture'] and before['config'] == after['config']
    assert report['resume_sha256'] == sha(args.before)
    assert before['weights_sha256'] == report['behavior_sha256'] == spec['model_sha256']
    assert after['weights_sha256'] == report['weights_sha256']
    assert after['actor_training'] == report['actor_training']
    previous_training=before.get('actor_training',dict(scope='all',learning_rate=before['config']['actor_lr']))
    actor_scope_preserved=previous_training==after['actor_training']
    assert actor_scope_preserved or report['actor_scope_fork'], 'Actor scope changed without explicit fork'
    assert report['sequence_retention_objective_fork'] or before.get('sequence_retention')==after.get('sequence_retention'), 'Sequence objective changed without explicit fork'
    retention = after['sequence_retention']
    assert retention == report['sequence_retention']
    assert retention['spec_sha256'] == sha(args.spec) and retention['version'] == spec['version']
    assert retention['behavior_sha256'] == before['weights_sha256'] and retention['weight'] > 0
    assert after['consumed_rollouts'] == before['consumed_rollouts'] + [report['rollout_sha256']]
    assert report['actor_steps'] > 0
    assert after['updates_completed'] == before['updates_completed'] + 1
    assert after['total_actor_steps'] == before['total_actor_steps'] + report['actor_steps']
    for name in ('sequence_retention_kl_before', 'sequence_retention_kl_after'):
        assert math.isfinite(report[name]) and report[name] >= -1e-8
    assert abs(report['sequence_retention_kl_before']) < 1e-8
    assert before['actor'].keys() == after['actor'].keys()
    changed = {}
    for name, old in before['actor'].items():
        new = after['actor'][name]
        assert old.is_cuda and new.is_cuda and old.shape == new.shape
        assert bool(torch.isfinite(new).all())
        changed[name] = int((old != new).sum())
    movement = aim = 0
    for name in ('base.head.weight', 'base.head.bias', 'base.residual.weight', 'base.residual.bias'):
        old, new = before['actor'][name], after['actor'][name]
        movement += int((old[:2] != new[:2]).sum())
        aim += int((old[2:4] != new[2:4]).sum()) + int((old[29:] != new[29:]).sum())
    encoder = sum(n for name, n in changed.items() if name.startswith('base.encoder.'))
    assert encoder > 0 and movement > 0 and aim > 0, 'Full policy update did not reach encoder/movement/aim'
    for module in ('actor', 'value'):
        old_opt, new_opt = before[module + '_optimizer'], after[module + '_optimizer']
        old_ids = [p for g in old_opt['param_groups'] for p in g['params']]
        new_ids = [p for g in new_opt['param_groups'] for p in g['params']]
        assert old_ids == new_ids
        for index, state in new_opt['state'].items():
            assert all(bool(torch.isfinite(v).all()) for v in state.values())
            if module == 'actor':
                old_step = old_opt['state'].get(index, {}).get('step', 0)
                assert float(state['step'] - old_step) == report['actor_steps']
    durable_json(args.out, dict(version='combat_full_checkpoint_audit_v1', device='cuda',
        before_checkpoint_sha256=sha(args.before), after_checkpoint_sha256=sha(args.after),
        report_sha256=sha(args.report), spec_sha256=sha(args.spec), actor_steps=report['actor_steps'],
        changed_actor_scalars=changed, changed_encoder_scalars=encoder,
        changed_movement_output_scalars=movement, changed_aim_output_scalars=aim,
        consumed_rollout_append_verified=True, optimizer_clocks_verified=True,
        actor_scope_preserved=actor_scope_preserved,actor_scope_fork=report['actor_scope_fork'],
        scope='Resumed full actor and own-history retention. Does not establish gameplay quality.'))


if __name__ == '__main__':
    main()
