"""Zero-column v1 -> vertical v2 migration, CUDA likelihood/KL and Adam audit."""
import argparse
import copy
import json
import pathlib

from ppo_recurrent import torch, nn, prepare_context, sequence, durable_json, durable_write
from migrate_combat_navigation_cuda import branch
from combat_sampled_motion_aim import VERSION, VERTICAL_VERSION, VERTICAL_CONTROL_VERSION, correction, validate
from combat_precision_head import probabilities
from combat_sequence_retention import policy_kl
from combat_target_head import availability
from combat_weapon_state_residual import optimizer
from fork_conditional_weapon_exploration_cuda import equal_tree
from process_combat_architecture_pool import read, sha


def main():
    ap = argparse.ArgumentParser()
    for name in ('source', 'data', 'out'):
        ap.add_argument('--'+name, type=pathlib.Path, required=True)
    ap.add_argument('--ablated', action='store_true')
    args = ap.parse_args()
    assert torch.cuda.is_available() and not args.out.exists()
    torch.set_default_device('cuda'); torch.set_num_threads(2); torch.manual_seed(20261021)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    pins = {}
    def pin(path, expected=None):
        path = pathlib.Path(path); digest = sha(path)
        assert expected is None or digest == expected
        pins[str(path.resolve())] = digest
        return path
    seal = read(pin(args.source/'complete.json'))
    for name, key in [('weights.json','weights_sha256'), ('checkpoint.pt','checkpoint_sha256'), ('report.json','report_sha256')]:
        pin(args.source/name, seal[key])
    model = read(args.source/'weights.json')
    assert model['sampled_motion_aim']['version'] == VERSION
    cp = torch.load(args.source/'checkpoint.pt', map_location='cuda', weights_only=True)
    assert cp['weights_sha256'] == seal['weights_sha256']
    updated = copy.deepcopy(model)
    updated['sampled_motion_aim']['version'] = VERTICAL_CONTROL_VERSION if args.ablated else VERTICAL_VERSION
    for row in updated['sampled_motion_aim']['layers'][0]['weight']:
        row.extend([0., 0., 0.])
    validate(updated)
    before, after = branch(model, 'actor'), branch(updated, 'actor')
    assert equal_tree(before.state_dict(), cp['actor'])
    assert equal_tree(branch(updated, 'value').state_dict(), cp['value'])
    padded_name = 'sampled_motion_aim_branch.0.weight'
    for name, tensor in before.state_dict().items():
        migrated = after.state_dict()[name]
        assert torch.equal(tensor, migrated[:, :125] if name == padded_name else migrated)
    meta = read(pin(args.data/'report.json'))
    assert meta['model_sha256'] == seal['weights_sha256']
    streams = {}
    for name in ('rollout', 'sequence'):
        path = pin(args.data/(name+'.jsonl'), meta[name+'_sha256'])
        streams[name] = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()]
    rows = streams['rollout']
    x = torch.tensor([r['features'] for r in rows], dtype=torch.float32, device='cuda')
    def samples(key, dtype):
        return torch.tensor([r['sample'].get(key, 0) for r in rows], dtype=dtype, device='cuda')
    z = samples('latent', torch.float32); attack = samples('attack', torch.float32)
    vertical, weapon, target, mode = [samples(k, torch.long) for k in ('vertical','weapon','target','aim_mode')]
    std = nn.Parameter(cp['log_std'].clone()); new_std = nn.Parameter(cp['log_std'].clone())
    with torch.no_grad():
        prepared = prepare_context(streams['sequence'], rows, 'cuda')
        old, error = sequence(before, prepared, check_states=True)
        new, new_error = sequence(after, prepared, check_states=True)
        torch.testing.assert_close(old, new, rtol=0, atol=1e-5)
        assert max(error, new_error) < 2e-5
        old_lp, _ = probabilities(old,std,z,attack,vertical,weapon,x,target,mode,before.sampled_motion_aim_branch)
        new_lp, _ = probabilities(new,new_std,z,attack,vertical,weapon,x,target,mode,after.sampled_motion_aim_branch)
        torch.testing.assert_close(new_lp,old_lp,rtol=0,atol=1e-5)
        native_error = float((new_lp-samples('log_probability',torch.float32)).abs().max())
        assert native_error < .003
        for v in range(3):
            selected = torch.full_like(vertical,v)
            torch.testing.assert_close(
                correction(before.sampled_motion_aim_branch,x,old,z[:,:2],target,mode),
                correction(after.sampled_motion_aim_branch,x,new,z[:,:2],target,mode,selected),rtol=0,atol=1e-6)

    # An explicit nonzero vertical-only path. Check the actual joint likelihood
    # and an independent closed-form categorical expectation for retention.
    probe = copy.deepcopy(after.sampled_motion_aim_branch)
    probe.motion_version = VERTICAL_VERSION
    neutral = copy.deepcopy(before.sampled_motion_aim_branch)
    with torch.no_grad():
        for module in (probe, neutral):
            for p in module.parameters(): p.zero_()
        for v in range(3):
            probe[0].weight[v,125+v] = 1
            probe[2].weight[v,v] = 1
            probe[4].weight[0,v] = v*.1
        expected = torch.where(target>0,vertical.float()*.1,torch.zeros_like(vertical,dtype=torch.float32))
        delta = correction(probe,x,new,z[:,:2],target,mode,vertical)
        torch.testing.assert_close(delta[:,0],expected,rtol=0,atol=1e-7)
        assert torch.equal(delta[:,1],torch.zeros_like(delta[:,1]))
        base_lp, _ = probabilities(new,new_std,z,attack,vertical,weapon,x,target,mode,neutral)
        probe_lp, _ = probabilities(new,new_std,z,attack,vertical,weapon,x,target,mode,probe)
        from combat_precision_head import selected_means
        means = selected_means(new,target,mode)[:,2]
        expected_lp = -.5*(((z[:,2]-means-expected)/new_std[2].exp()).square()-((z[:,2]-means)/new_std[2].exp()).square())
        torch.testing.assert_close(probe_lp-base_lp,expected_lp,rtol=1e-4,atol=.003)
        none = torch.zeros_like(target)
        assert torch.equal(correction(probe,x,new,z[:,:2],none,mode,vertical),torch.zeros_like(delta))
        same = policy_kl(new,new_std,new,new_std,x,after.sampled_motion_aim_branch,after.sampled_motion_aim_branch,z[:,:2].detach())
        assert float(same.abs().max()) < 1e-8
    kl = policy_kl(new,new_std,new,new_std,x,probe,neutral,z[:,:2].detach())
    with torch.no_grad():
        none_probability = new.double()[:,20:29].masked_fill(~availability(x),float('-inf')).softmax(-1)[:,0]
        costs = .5*torch.tensor([0.,.1,.2],device='cuda',dtype=torch.float64).square()/new_std[2].double().exp().square()
        expected_kl = (1-none_probability)*(new.double()[:,5:8].softmax(-1)*costs).sum(-1)
        torch.testing.assert_close(kl,expected_kl,rtol=1e-5,atol=1e-7)
        assert float(kl.max()) > .001
    kl.mean().backward()
    assert probe[0].weight.grad is not None and float(probe[0].weight.grad[:,125:].norm()) > 0

    old_opt = optimizer(before,std,cp['config']['actor_lr'],cp['actor_training']['weapon_state_residual']['learning_rate'])
    new_opt = optimizer(after,new_std,cp['config']['actor_lr'],cp['actor_training']['weapon_state_residual']['learning_rate'])
    old_opt.load_state_dict(copy.deepcopy(cp['actor_optimizer']))
    old_named = dict(before.named_parameters()); old_named['@log_std'] = std
    new_named = dict(after.named_parameters()); new_named['@log_std'] = new_std
    for name, old_parameter in old_named.items():
        state = copy.deepcopy(old_opt.state.get(old_parameter,{}))
        if name == padded_name:
            for key, value in state.items():
                if torch.is_tensor(value) and value.shape == old_parameter.shape:
                    state[key] = torch.cat((value,torch.zeros((64,3),device=value.device,dtype=value.dtype)),1)
        new_opt.state[new_named[name]] = state
        for key, old_value in old_opt.state.get(old_parameter,{}).items():
            value = state[key]
            if name == padded_name and torch.is_tensor(old_value) and old_value.shape == old_parameter.shape:
                assert torch.equal(value[:,:125],old_value) and not bool(value[:,125:].any())
            else: assert equal_tree(value,old_value)
    for new_group, old_group in zip(new_opt.param_groups,old_opt.param_groups):
        for key, value in old_group.items():
            if key != 'params': new_group[key] = copy.deepcopy(value)
    args.out.mkdir(parents=True)
    durable_json(args.out/'weights.json',updated)
    receipt = dict(state='complete',device='cuda',version=updated['sampled_motion_aim']['version'],
        history_source=str(args.source.resolve()),history_source_weights_sha256=seal['weights_sha256'],
        zero_column_migration=True,history_rows=len(rows),history_raw_max_error=float((new-old).abs().max()),
        native_log_probability_max_error=native_error,all_vertical_means_equivalent=True,
        nonzero_vertical_likelihood_verified=True,exact_vertical_retention_verified=True,
        none_target_neutral=True,optimizer_states_preserved=True,training_performed=False,promotion=False,
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__))
    fork = copy.deepcopy(cp)
    fork.update(actor=after.state_dict(),weights_sha256=sha(args.out/'weights.json'),actor_optimizer=new_opt.state_dict(),sequence_retention=None)
    fork['model_migrations'] = cp.get('model_migrations',[])+[copy.deepcopy(receipt)]
    durable_write(args.out/'checkpoint.pt',lambda f:torch.save(fork,f))
    assert equal_tree(torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True),fork)
    assert equal_tree(branch(updated,'actor').state_dict(),fork['actor'])
    for path,digest in pins.items(): assert sha(path) == digest
    durable_json(args.out/'report.json',receipt)
    durable_json(args.out/'complete.json',dict(version='combat_update_complete_v1',weights_sha256=sha(args.out/'weights.json'),checkpoint_sha256=sha(args.out/'checkpoint.pt'),report_sha256=sha(args.out/'report.json')))
    print(json.dumps({k:v for k,v in receipt.items() if k != 'inputs'}))


if __name__ == '__main__': main()
