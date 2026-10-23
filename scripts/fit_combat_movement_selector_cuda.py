"""Fixed-budget offline movement selectors on frozen full-state temporal embeddings."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil

from evaluate_registered_combat_models import read, save, sha
from ppo_combat import torch, nn, training_devices, layers
from ppo_recurrent import prepare_context, sequence
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model
from fork_conditional_weapon_exploration_cuda import equal_tree


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'source', 'protocol', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or expected == digest, str(path)
        pins[str(path)] = digest
        return path

    p = read(pin(a.protocol))
    assert p['version'] == 'combat_movement_selector_fit_v1'
    ds_seal = read(pin(a.dataset/'complete.json'))
    ds = read(pin(a.dataset/'report.json', ds_seal['report_sha256']))
    assert sha(a.dataset/'report.json') == p['dataset_report_sha256']
    assert ds['gate_passed'] and not ds['training_performed'] and not ds['ordinary_ppo_eligible']
    data = read(pin(a.dataset/'records.json', p['records_sha256']))['records']
    assert ds['records_sha256'] == p['records_sha256']
    for path, digest in ds['inputs'].items():
        pin(path, digest)
    source_seal = read(pin(a.source/'complete.json'))
    model = read(pin(a.source/'weights.json', source_seal['weights_sha256']))
    pin(a.source/'checkpoint.pt', source_seal['checkpoint_sha256'])
    pin(a.source/'report.json', source_seal['report_sha256'])
    assert source_seal['weights_sha256'] == p['source_weights_sha256'] == ds['source_weights_sha256']
    assert model['attention']['heads'] == 4 and model['attention']['window'] == 32
    train = [i for i,r in enumerate(data) if r['split'] == 'train']
    dev = [i for i,r in enumerate(data) if r['split'] == 'development']
    assert len(train) == ds['counts']['train_contexts'] and len(dev) == ds['counts']['development_contexts']
    assert not {data[i]['seed'] for i in train} & {data[i]['seed'] for i in dev}
    training_devices()
    torch.set_default_device('cuda')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    actor = branch(base_model(model, allow_shared=True), 'actor').requires_grad_(False)
    before = copy.deepcopy(actor.state_dict())
    embeddings, memory_errors, mean_errors = [], [], []
    a.out.mkdir(parents=True)
    shutil.copy2(__file__, a.out/Path(__file__).name)
    shutil.copy2(a.protocol, a.out/'protocol.json')
    candidates = p['candidates']
    assert candidates == ['control','left','right','forward','backward','near_stationary']
    for i, record in enumerate(data):
        save(a.out/'progress.json', dict(stage='cuda-frozen-embedding', context=i, contexts=len(data)))
        context = record['context']
        folder = Path(context['directory'])/'cuda'
        meta = read(pin(folder/'report.json'))
        assert meta['observation_timing_contract'] == 'frozen_world_v1'
        rows = list(map(json.loads, pin(folder/'rollout.jsonl', meta['rollout_sha256']).open(encoding='utf-8-sig')))
        context_rows = list(map(json.loads, pin(folder/'sequence.jsonl', meta['sequence_sha256']).open(encoding='utf-8-sig')))
        prepared = prepare_context(context_rows, rows, 'cuda')
        head_inputs, temporal_inputs = [], []
        hooks = [actor.base.head.register_forward_pre_hook(lambda m,args:head_inputs.append(args[0].detach())),
                 actor.base.residual.register_forward_pre_hook(lambda m,args:temporal_inputs.append(args[0].detach()))]
        with torch.no_grad():
            try:
                raw, error = sequence(actor, prepared, 32, True, 'actor')
            finally:
                for hook in hooks:
                    hook.remove()
            assert len(head_inputs) == len(temporal_inputs) == 1 and error < 2e-5
            _, positions, inverse, _ = prepared
            h = torch.cat((head_inputs[0], temporal_inputs[0]), -1)
            h = h[positions[:,0], positions[:,1]][inverse]
            assert h.shape == (len(rows),128)
            at = next(j for j,r in enumerate(rows) if r['frame'] == context['frame'])
            assert rows[at]['seed'] == record['seed']
            embeddings.append(h[at].clone())
            proof = read(pin(record['proof'], record['proof_sha256']))
            assert proof['baseline_full_exact'] and proof['excluded'] == 0
            for path, digest in proof['inputs'].items():
                pin(path, digest)
            # Accepted zero-control: movement mean = recorded source latent minus reused additive Gaussian draw.
            zero_path = next(Path(path) for path in proof['inputs'] if Path(path).name == 'weights.json' and Path(path).parent.name == 'control')
            zero_model = read(zero_path)
            case_report = next(Path(path) for path in proof['inputs'] if Path(path).name == 'cuda-verification.json'
                and Path(path).parent.parent.parent.resolve() == a.dataset.parent.resolve())
            zero_meta = read(case_report.with_name('report.json'))
            zero_rows = list(map(json.loads, pin(case_report.with_name('rollout.jsonl'), zero_meta['rollout_sha256']).open(encoding='utf-8-sig')))
            zero = next(r for r in zero_rows if r['frame'] == context['frame'])
            assert zero_model['diagnostic_movement_probe']['mean_shift'] == [0.,0.]
            mean = torch.tensor(zero['sample']['aim_candidate']['source_latent'][:2],device='cuda')
            mean -= torch.tensor(zero['sample']['movement_probe']['gaussian_noise'][:2],device='cuda')
            mean_error = float((raw[at,:2]-mean).abs().max())
            assert mean_error < 2e-5
            memory_errors.append(error)
            mean_errors.append(mean_error)
    x = torch.stack(embeddings).detach()
    ti, di = torch.tensor(train,device='cuda'), torch.tensor(dev,device='cuda')
    mean, std = x[ti].mean(0), x[ti].std(0,unbiased=False).clamp_min(.1)
    x = ((x-mean)/std).clamp(-10,10).detach()
    valid = torch.zeros((len(data),6),device='cuda',dtype=torch.bool)
    useful, harmful, utility = [torch.zeros((len(data),6),device='cuda') for _ in range(3)]
    for i,r in enumerate(data):
        zero = next(v for v in r['labels'] if v['label'] == 'control')
        for v in r['labels']:
            j = candidates.index(v['label'])
            valid[i,j] = True
            useful[i,j] = float(v['useful'])
            harmful[i,j] = float((zero['goal'] and not zero['death'] and not(v['goal'] and not v['death'])) or
                (zero['goal'] and v['goal'] and not zero['death'] and not v['death'] and v['kills'] == zero['kills'] and v['received']-zero['received'] >= 5))
            utility[i,j] = 4*float(v['goal'] and not v['death'])+.25*v['kills']-v['received']/100.-float(v['death'])
    assert bool(valid[:,0].all())
    pair_masks = valid[:,:,None] & valid[:,None,:] & (utility[:,:,None]-utility[:,None,:] >= .05)
    pair_count = pair_masks.sum((1,2))
    assert bool((pair_count[ti]>0).any())
    settings = p['optimizer']
    results = []
    for name, widths in p['architectures'].items():
        torch.manual_seed(20261023)
        net = (nn.Linear(128,6,device='cuda') if len(widths)==2 else nn.Sequential(
            nn.Linear(128,8,device='cuda'),nn.ReLU(),nn.Linear(8,6,device='cuda')))
        last = net if isinstance(net,nn.Linear) else net[-1]
        with torch.no_grad():
            last.weight.zero_();last.bias.zero_();last.bias[0]=math.log(495.)
        optimizer = torch.optim.Adam(net.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])

        def probs():
            scores = net(x).clamp(-settings['score_clip'],settings['score_clip'])
            logp = scores.masked_fill(~valid,-torch.inf).log_softmax(-1)
            return scores,logp,logp.exp()

        def loss():
            scores,logp,_ = probs()
            pair = nn.functional.softplus(scores[:,None,:]-scores[:,:,None])
            per_context = (pair*pair_masks).sum((1,2))/pair_count.clamp_min(1)
            return per_context[ti[pair_count[ti]>0]].mean()+.05*(-logp[ti,0]).mean()

        start = float(loss().detach())
        for step in range(settings['steps']):
            optimizer.zero_grad(set_to_none=True)
            objective = loss()
            assert bool(torch.isfinite(objective))
            objective.backward()
            nn.utils.clip_grad_norm_(net.parameters(),settings['clip_norm'],error_if_nonfinite=True)
            optimizer.step()
        with torch.no_grad():
            _,_,probability = probs()
            constant = probability[ti].mean(0)
            controls = constant[None,:]*valid
            controls = controls/controls.sum(-1,keepdim=True)
            delta = utility-utility[:,:1]

            def metrics(ids, q):
                return dict(useful_mass=float((q[ids]*useful[ids]).sum(-1).mean()),
                    harmful_mass=float((q[ids]*harmful[ids]).sum(-1).mean()),
                    observed_utility_delta=float((q[ids]*delta[ids]).sum(-1).mean()),
                    control_mass=float(q[ids,0].mean()))

            development, control_dev = metrics(di,probability),metrics(di,controls)
            per_gain = ((probability[di]-controls[di])*useful[di]).sum(-1)
            gates = dict(useful=development['useful_mass'] >= control_dev['useful_mass']+p['gate']['development_useful_mass_over_constant'],
                harmful=development['harmful_mass'] <= control_dev['harmful_mass']+p['gate']['development_harmful_mass_over_constant_max'],
                utility=development['observed_utility_delta'] >= control_dev['observed_utility_delta']+p['gate']['development_observed_utility_delta_over_constant'],
                contexts=int((per_gain>0).sum()) >= p['gate']['minimum_development_contexts_with_useful_mass_gain'])
            folder = a.out/name
            folder.mkdir()
            exported = layers(nn.Sequential(net)) if isinstance(net,nn.Linear) else layers(net)
            save(folder/'head.json', dict(version='combat_movement_selector_offline_v1', source_weights_sha256=p['source_weights_sha256'],
                embedding='frozen_actor_encoder64_attention_output64_v1', mean=mean.tolist(),std=std.tolist(),
                input_clip=10, score_clip=settings['score_clip'],candidates=candidates,layers=exported,
                ordinary_policy_export=False, runtime_implemented=False))
            save(folder/'probabilities.json', [dict(seed=r['seed'],split=r['split'],context_id=r['context']['context_id'],
                probabilities=probability[i].tolist(),constant=controls[i].tolist(),available=valid[i].tolist()) for i,r in enumerate(data)])
            result = dict(name=name,parameters=sum(v.numel() for v in net.parameters()),steps=settings['steps'],
                loss_before=start,loss_after=float(loss()),train=metrics(ti,probability),development=development,
                constant_development=control_dev,gates=gates,gate_passed=all(gates.values()),head_sha256=sha(folder/'head.json'))
            results.append(result)
    assert equal_tree(actor.state_dict(),before)
    for path, digest in pins.items():
        assert sha(path) == digest, path
    passed = sorted([r for r in results if r['gate_passed']],key=lambda r:(
        -(r['development']['useful_mass']-r['constant_development']['useful_mass']),r['development']['harmful_mass'],r['parameters']))
    report = dict(state='complete',version='combat_movement_selector_offline_fit_v1',device='cuda',results=results,
        selected=passed[0]['name'] if passed else None,gate_passed=bool(passed),train_contexts=len(train),development_contexts=len(dev),
        source_neural_parameters_exact=True,source_checkpoint_unchanged=True,max_memory_error=max(memory_errors),
        max_movement_mean_error=max(mean_errors),inputs=pins,script_sha256=sha(__file__),
        fitting_performed=True,new_games=False,ordinary_policy_export=False,runtime_implemented=False,promotion=False,
        scope='Fixed offline endpoints on observed one-frame conditional continuations. No expected Q, no battle-quality or superiority claim. Native/CUDA runtime contract and independent games required before policy acceptance.')
    save(a.out/'report.json',report)
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    save(a.out/'progress.json',dict(stage='complete',selected=report['selected'],gate_passed=report['gate_passed']))
    print(dict(selected=report['selected'],results=results))


if __name__ == '__main__':
    main()
