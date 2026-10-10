"""Exact per-head policy change on pinned real histories; CUDA inference only."""
import argparse
import json
import pathlib

from ppo_recurrent import read, sha, training_devices, torch, Recurrent, prepare_context, sequence
from combat_attention import CausalAttention
from combat_target_head import availability, distribution as targets
from combat_precision_head import probabilities as precision_probabilities
from combat_target_head import probabilities as target_probabilities
from combat_weapon_head import feature_mask
from combat_sequence_retention import normal_kl, categorical_kl, policy_kl
from process_combat_architecture_pool import save


def components(before, after, old_std, std, features):
    assert all(v.is_cuda for v in (before, after, old_std, std, features))
    old, new = before.double(), after.double()
    old_std, std = old_std.double(), std.double()
    valid = availability(features)
    probabilities = torch.softmax(old[:,20:29].masked_fill(~valid, float('-inf')), -1)
    result = dict(
        movement=normal_kl(old[:,:2], new[:,:2], old_std[:2], std[:2]),
        attack=torch.distributions.kl_divergence(torch.distributions.Bernoulli(logits=old[:,4]), torch.distributions.Bernoulli(logits=new[:,4])),
        vertical=categorical_kl(old[:,5:8], new[:,5:8]),
        weapon=categorical_kl(old[:,8:20], new[:,8:20], feature_mask(features[:,:845])),
        target=categorical_kl(old[:,20:29], new[:,20:29], valid))
    coarse_old = torch.cat((old[:,None,2:4], old[:,29:45].reshape(-1,8,2)), 1)
    coarse_new = torch.cat((new[:,None,2:4], new[:,29:45].reshape(-1,8,2)), 1)
    aim = normal_kl(coarse_old, coarse_new, old_std[2:], std[2:])
    mode = torch.zeros_like(probabilities)
    if old.shape[1] == 81:
        old_mode, new_mode = old[:,63:81].reshape(-1,9,2), new[:,63:81].reshape(-1,9,2)
        mode = categorical_kl(old_mode, new_mode)
        weights = old_mode.softmax(-1)
        fine = normal_kl(old[:,45:63].reshape(-1,9,2), new[:,45:63].reshape(-1,9,2), old_std[2:], std[2:])
        aim = weights[:,:,0] * aim + weights[:,:,1] * fine
    result['aim_mode'] = (probabilities * mode).sum(-1)
    result['aim'] = (probabilities * aim).sum(-1)
    for values in result.values():
        assert bool(torch.isfinite(values).all()) and float(values.min()) >= -1e-9
    total = sum(result.values())
    reference = policy_kl(after, std, before, old_std, features)
    assert float((total - reference).abs().max()) < 1e-9, 'Head sum differs from full policy KL'
    return result, total


def main():
    parser = argparse.ArgumentParser()
    for name in ('before', 'after', 'data', 'out'):
        parser.add_argument('--'+name, type=pathlib.Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'Fresh audit output required'
    training_devices()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    before, after, meta = read(args.before), read(args.after), read(args.data / 'report.json')
    assert sha(args.before) == meta['model_sha256']
    assert before['feature_version'] == after['feature_version'] == meta['feature_version']
    assert before.get('target_head') == after.get('target_head') and before.get('target_head')
    assert before.get('aim_mode_head') == after.get('aim_mode_head')
    assert bool(before.get('spatial_aim')) == bool(after.get('spatial_aim'))
    key = next(k for k in ('memory', 'attention') if before.get(k))
    assert before[key]['version'] == after[key]['version'] == meta['recurrent_version']
    if key == 'attention':
        assert before[key]['window'] == after[key]['window'] and before[key]['heads'] == after[key]['heads']
    for path, digest in meta['source_sha256'].items():
        assert sha(pathlib.Path(path)) == digest, 'Changed native source: '+path
    for name in ('rollout', 'sequence'):
        assert sha(args.data / (name+'.jsonl')) == meta[name+'_sha256']
    rows = [json.loads(line) for line in (args.data / 'rollout.jsonl').read_text().splitlines()]
    context = [json.loads(line) for line in (args.data / 'sequence.jsonl').read_text().splitlines()]
    assert len(rows) == meta['rows'] and len(context) == meta['sequence_rows']
    prepared = prepare_context(context, rows, 'cuda')
    features = torch.tensor([r['features'] for r in rows], dtype=torch.float32, device='cuda')
    def build(model):
        spec = model[key]
        module = Recurrent(model['actor'], spec['actor']) if key == 'memory' else CausalAttention(model['actor'], spec['actor'], spec['heads'], spec['window'])
        if model.get('spatial_aim'):
            from combat_spatial_aim import wrap
            module = wrap(module, model)
        return module.to('cuda').eval()
    with torch.no_grad():
        old, state_error = sequence(build(before), prepared, check_states=True)
        new, _ = sequence(build(after), prepared)
        old_std = torch.tensor(before['log_std'], device='cuda')
        std = torch.tensor(after['log_std'], device='cuda')
        sample = lambda name, dtype: torch.tensor([r['sample'].get(name,0) if name in ('weapon','target','aim_mode') else r['sample'][name] for r in rows], dtype=dtype, device='cuda')
        likelihood = precision_probabilities if before.get('aim_mode_head') else target_probabilities
        likelihood_args = [old, old_std, sample('latent', torch.float32), sample('attack', torch.float32), sample('vertical', torch.long), sample('weapon', torch.long), features, sample('target', torch.long)]
        if before.get('aim_mode_head'):
            likelihood_args.append(sample('aim_mode', torch.long))
        lp, _ = likelihood(*likelihood_args)
        error = float((lp-sample('log_probability', torch.float32)).abs().max())
        assert error < .003 and state_error < 2e-5, (error, state_error)
        terms, total = components(old, new, old_std, std, features)
        choice_before, choice_after = targets(old[:,:45], features), targets(new[:,:45], features)
        previous = features[:,845:854]
        known_previous = previous[:,1:].sum(-1) == 1
        previous_probability = lambda choice: float((choice.probs * previous).sum(-1)[known_previous].mean()) if bool(known_previous.any()) else None
        result = dict(version='combat_head_learning_audit_v1', device='cuda', rows=len(rows),
                      before_sha256=sha(args.before), after_sha256=sha(args.after), data_sha256=sha(args.data / 'report.json'),
                      script_sha256=sha(pathlib.Path(__file__)), old_log_probability_max_error=error, old_actor_memory_max_error=state_error,
                      exact_policy_kl_mean=float(total.mean()),
                      heads={name:dict(kl_mean=float(value.mean()), kl_max=float(value.max())) for name,value in terms.items()},
                      target_entropy_before=float(choice_before.entropy().mean()), target_entropy_after=float(choice_after.entropy().mean()),
                      target_total_variation_mean=float((choice_before.probs-choice_after.probs).abs().sum(-1).mean()/2),
                      target_argmax_changed_fraction=float((choice_before.probs.argmax(-1)!=choice_after.probs.argmax(-1)).float().mean()),
                      known_previous_target_rows=int(known_previous.sum()),
                      previous_target_probability_before=previous_probability(choice_before), previous_target_probability_after=previous_probability(choice_after),
                      scope='Exact KL(before||after), weighted over all available before-target/mode branches on actual training histories. No gradients or held-out fitting. Tiny target KL is not proof of a causal training bottleneck; trainer uses sampled approximate joint KL.')
    save(args.out, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
