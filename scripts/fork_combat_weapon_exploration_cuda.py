"""Explicit weapon-only exploration fork of a sealed attention checkpoint.

Scale both additive weapon outputs, clear only their Adam moment rows, retain
parameter clocks/history/RNG. A new own-policy training corpus is mandatory.
CUDA verification uses real complete histories; it does not train on them.
"""
import argparse
import copy
import json
import math
import pathlib

from ppo_recurrent import torch, read, sha, durable_json, durable_write, prepare_context, sequence
from migrate_combat_navigation_cuda import branch, validate
from migrate_combat_navigation_checkpoint_cuda import verify
from combat_weapon_head import feature_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=pathlib.Path, required=True)
    parser.add_argument('--data', type=pathlib.Path, required=True)
    parser.add_argument('--execution', type=pathlib.Path, required=True)
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available() and not args.out.exists()
    assert math.isfinite(args.temperature) and 1 < args.temperature <= 4
    torch.set_default_device('cuda'); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    seal = read(args.source/'complete.json')
    for filename, field in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):
        assert sha(args.source/filename) == seal[field], 'Broken source seal'
    model = read(args.source/'weights.json'); validate(model)
    assert model.get('attention') and not model.get('memory') and not model['deterministic']
    assert model['feature_version'] == 'combat_features_v10' and len(model['actor'][-1]['bias']) == 81
    cp = torch.load(args.source/'checkpoint.pt', map_location='cuda', weights_only=True)
    assert cp['weights_sha256'] == seal['weights_sha256']
    original_actor, original_value = branch(model,'actor'), branch(model,'value')
    for role, module in (('actor',original_actor),('value',original_value)):
        assert module.state_dict().keys() == cp[role].keys()
        assert all(torch.equal(v,cp[role][k]) for k,v in module.state_dict().items())
    meta = read(args.data/'report.json'); assert meta['model_sha256'] == seal['weights_sha256']
    for filename in ('rollout','sequence'):
        assert sha(args.data/(filename+'.jsonl')) == meta[filename+'_sha256']
    with (args.data/'rollout.jsonl').open(encoding='utf-8-sig') as stream:
        rows = [json.loads(line) for line in stream]
    with (args.data/'sequence.jsonl').open(encoding='utf-8-sig') as stream:
        context = [json.loads(line) for line in stream]
    prepared = prepare_context(context, rows, 'cuda')
    features = torch.tensor([r['features'] for r in rows], dtype=torch.float32, device='cuda')
    execution = read(args.execution)
    releases = {c['seed']:c['network_release_frame'] for c in execution['cases']}
    indices = [i for i,r in enumerate(rows) if r['seed'] in releases and r['frame'] == releases[r['seed']]]
    assert len(indices) == len(releases) == 32 and all(rows[i]['features'][1132] == 1 for i in indices)
    new = copy.deepcopy(model)
    for layer in (new['actor'][-1],new['attention']['actor']['residual']):
        for field in ('weight','bias'):
            tensor = torch.tensor(layer[field],dtype=torch.float32,device='cuda')
            tensor[8:20] /= args.temperature
            layer[field] = tensor.cpu().tolist()
    validate(new)
    fork = copy.deepcopy(cp)
    modified = {'base.head.weight','base.head.bias','base.residual.weight','base.residual.bias'}
    new_actor = branch(new,'actor')
    assert modified <= set(cp['actor'])
    fork['actor'] = copy.deepcopy(new_actor.state_dict())
    names = [name for name,_ in original_actor.named_parameters()] + ['log_std']
    groups = fork['actor_optimizer']['param_groups']; assert len(groups) == 1
    identifiers = groups[0]['params']; assert len(identifiers) == len(names)
    for identifier, name in zip(identifiers,names):
        if name not in modified: continue
        moments = fork['actor_optimizer']['state'][identifier]
        assert set(moments) <= {'step','exp_avg','exp_avg_sq','max_exp_avg_sq'}
        for field in set(moments)-{'step'}:
            moments[field][8:20].zero_()
    for name, old in cp['actor'].items():
        after = fork['actor'][name]
        if name in modified:
            assert torch.equal(old[:8],after[:8]) and torch.equal(old[20:],after[20:])
            assert torch.equal(old[8:20]/args.temperature,after[8:20])
        else: assert torch.equal(old,after)
    for role in ('actor','value'):
        for identifier, old in cp[role+'_optimizer']['state'].items():
            name = names[identifiers.index(identifier)] if role == 'actor' else ''
            for field, before in old.items():
                after = fork[role+'_optimizer']['state'][identifier][field]
                if role == 'actor' and name in modified and field != 'step':
                    assert torch.equal(before[:8],after[:8]) and torch.equal(before[20:],after[20:])
                    assert not bool(after[8:20].any())
                else: assert torch.equal(before,after)
    with torch.no_grad():
        before, before_error = sequence(original_actor,prepared,check_states=True)
        after, after_error = sequence(new_actor,prepared,check_states=True)
        assert max(before_error,after_error) < 2e-5
        untouched = torch.cat((before[:,:8]-after[:,:8],before[:,20:]-after[:,20:]),-1)
        assert float(untouched.abs().max()) == 0
        weapon_error = float((before[:,8:20]/args.temperature-after[:,8:20]).abs().max())
        assert weapon_error < 2e-5
        valid = feature_mask(features[:,:845])
        selected = torch.tensor(indices,dtype=torch.long,device='cuda')
        probabilities = []
        for raw in (before,after):
            p = raw[:,8:20].double().masked_fill(~valid,-torch.inf).softmax(-1)[selected]
            probabilities.append(dict(keep=float(p[:,0].mean()),supported=float(p[:,1:5].sum(-1).mean()),
                entropy=float(-(p*p.clamp_min(1e-300).log()).sum(-1).mean())))
        assert probabilities[1]['entropy'] > probabilities[0]['entropy']
    previous = copy.deepcopy(fork.get('sequence_retention')); fork['sequence_retention'] = None
    receipt = dict(version='combat_weapon_exploration_fork_v1', device='cuda',temperature=args.temperature,
        source_weights_sha256=seal['weights_sha256'],source_checkpoint_sha256=seal['checkpoint_sha256'],
        data_report_sha256=sha(args.data/'report.json'),execution_sha256=sha(args.execution),
        real_history_rows=len(rows),initial_cases=len(indices),initial_probabilities=probabilities,
        unchanged_outputs_max_error=0,weapon_scaled_outputs_max_error=weapon_error,
        state_max_error=max(before_error,after_error),modified_parameters=sorted(modified),
        optimizer_policy='Clear affected weapon moment rows only; retain shared per-parameter step clocks and all other moments.',
        previous_sequence_retention=previous,training_performed=False,persisted_optimizer_steps=0,
        scope='Explicit exploration initialization, not PPO improvement or rollout equivalence. Archive old sequence objective; fresh own train-policy samples required.')
    args.out.mkdir(parents=True)
    durable_json(args.out/'weights.json',new); fork['weights_sha256'] = sha(args.out/'weights.json')
    receipt['weights_sha256'] = fork['weights_sha256']
    fork['model_migrations'] = cp.get('model_migrations',[]) + [copy.deepcopy(receipt)]
    durable_write(args.out/'checkpoint.pt',lambda stream: torch.save(fork,stream))
    restored = torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    checks = verify(copy.deepcopy(restored),new)
    for field in ('config','consumed_rollouts','updates_completed','total_actor_steps','anchor_sha256','bank_sha256','actor_training'):
        assert restored[field] == cp[field]
    assert torch.equal(restored['rng'],cp['rng']) and torch.equal(restored['log_std'],cp['log_std'])
    assert all(torch.equal(restored['value'][k],v) for k,v in cp['value'].items())
    assert len(restored['cuda_rng']) == len(cp['cuda_rng'])
    assert all(torch.equal(a,b) for a,b in zip(restored['cuda_rng'],cp['cuda_rng']))
    assert sha(args.source/'weights.json') == seal['weights_sha256'] and sha(args.source/'checkpoint.pt') == seal['checkpoint_sha256']
    receipt.update(checkpoint_sha256=sha(args.out/'checkpoint.pt'),optimizer_checks=checks,persisted_restoration_verified=True)
    durable_json(args.out/'report.json',receipt)
    durable_json(args.out/'complete.json',dict(weights_sha256=receipt['weights_sha256'],checkpoint_sha256=receipt['checkpoint_sha256'],report_sha256=sha(args.out/'report.json')))
    print(json.dumps({k:receipt[k] for k in ('device','temperature','real_history_rows','initial_probabilities','unchanged_outputs_max_error','weapon_scaled_outputs_max_error')}))


if __name__ == '__main__': main()
