"""CUDA contract checks on pinned own-TRAIN history; no native outcome claim."""
import argparse
import copy
import json
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha
from ppo_combat import torch, training_devices
from ppo_recurrent import prepare_context, sequence
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model
from combat_weapon_probe import VERSION, validate, adjust_log_probability


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); assert not a.out.exists()
    selected = read(a.selection)
    model = read(selected['source_model'])
    assert sha(selected['source_model']) == selected['source_weights_sha256']
    paths = selected['inputs']
    rollout = next(Path(p) for p in paths if p.endswith('/rollout.jsonl') or p.endswith('\\rollout.jsonl'))
    history = next(Path(p) for p in paths if p.endswith('/sequence.jsonl') or p.endswith('\\sequence.jsonl'))
    assert sha(rollout) == paths[str(rollout)] and sha(history) == paths[str(history)]
    rows = [json.loads(line) for line in rollout.open(encoding='utf-8-sig')]
    histories = [json.loads(line) for line in history.open(encoding='utf-8-sig')]
    training_devices(); torch.set_default_device('cuda'); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    actor = branch(base_model(model, allow_shared=True, allow_target_sampling=True), 'actor').requires_grad_(False)
    states = copy.deepcopy(actor.state_dict())
    context = prepare_context(histories, rows, 'cuda')
    with torch.inference_mode():
        raw, memory_error = sequence(actor, context, 32, True, 'actor')
        assert raw.is_cuda and memory_error < 2e-5
        x = torch.tensor([r['features'] for r in rows], device='cuda', dtype=torch.float32)
        weapon = torch.tensor([r['sample'].get('weapon', 0) for r in rows], device='cuda', dtype=torch.long)
        # One case per call: the same network frame occurs independently in
        # several episodes, and is never a cross-episode intervention index.
        checked = []
        for case in selected['cases']:
            indexes = [i for i, row in enumerate(rows) if row['seed'] == case['seed'] and row['frame'] in (98, 99, 100)]
            assert len(indexes) == 3
            ids = torch.tensor(indexes, device='cuda', dtype=torch.long)
            frames = torch.tensor([rows[i]['frame'] for i in indexes], device='cuda', dtype=torch.long)
            r, features = raw[ids], x[ids]
            chosen = weapon[ids].clone(); chosen[frames == 98] = 2
            logits = r[:, 8:20].masked_fill(~features[:, 833:845].bool(), float('-inf'))
            old_lp = torch.distributions.Categorical(logits=logits).log_prob(chosen)
            others = torch.full_like(old_lp, -3.25)
            lp = old_lp + others
            for mode in ('identity', 'force'):
                probe = copy.deepcopy(model)
                probe['diagnostic_weapon_probe'] = dict(version=VERSION, frame=98, mode=mode,
                                                       expected_equipped_weapon=1, forced_weapon=2)
                validate(probe)
                try:
                    base_model(probe, allow_shared=True, allow_target_sampling=True)
                except AssertionError:
                    pass
                else:
                    raise AssertionError('Diagnostic model entered ordinary scorer')
                clean = base_model(probe, allow_probe=True, allow_shared=True, allow_target_sampling=True)
                original_clean = base_model(model, allow_shared=True, allow_target_sampling=True)
                assert clean == original_clean, 'Diagnostic header modified network specification'
                result = adjust_log_probability(probe, frames, r, features, chosen, lp)
                assert torch.equal(result[frames != 98], lp[frames != 98])
                if mode == 'identity':
                    assert result is lp and torch.equal(result, lp)
                else:
                    assert float((result[frames == 98] - others[frames == 98]).abs().max()) < 1e-5
            checked.append({'map': case['map'], 'seed': case['seed'], 'rows': len(indexes)})
    assert all(torch.equal(states[k], v) and v.is_cuda for k, v in actor.state_dict().items())
    assert sha(selected['source_model']) == selected['source_weights_sha256']
    a.out.mkdir(parents=True)
    report = dict(state='passed', device='cuda', checked=checked, memory_error=memory_error,
                  identity_log_probability_exact=True, outside_frame_log_probability_exact=True,
                  forced_category_log_probability_zero=True, source_neural_state_unchanged=True,
                  ordinary_scoring_rejected=True, training_performed=False, native_games_run=0,
                  source_weights_sha256=selected['source_weights_sha256'], selection_sha256=sha(a.selection),
                  script_sha256=sha(__file__), helper_sha256=sha(Path(__file__).with_name('combat_weapon_probe.py')),
                  scope='Numerical intervention contract on captured history; synthetic changed weapon category, not new native proof.')
    save(a.out/'report.json', report); save(a.out/'complete.json', dict(state='complete', report_sha256=sha(a.out/'report.json')))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
