"""Weapon-only clipped policy gradient on fresh verified H-JEPA TRAIN rollouts.

Finite first-life episodes use a leave-one-stream-out return baseline per native
world. No diagnostic preference labels, critic, demonstrations, or viewed tests.
"""
import argparse
import collections
import copy
import json
import math
import pathlib
import torch
from torch import nn
from train_combat_hjepa import export, save, sha, serialized_forward
from train_combat_hjepa_branch_preferences import network


VERSION = 'hjepa_weapon_onpolicy_update_v2'
ALGORITHM = dict(return_version='first_life_win100_self1_v1', baseline='leave_one_stream_out',
                 advantage_scale=100, horizon=300, learning_rate=1e-4, clip=.1,
                 gradient_clip=1., max_batch_steps=4, max_behavior_kl=.02)
RTG_VERSION = 'hjepa_weapon_onpolicy_update_v3'
RTG_ALGORITHM = dict(ALGORITHM,baseline='leave_one_stream_time_return_to_go_v1')
NAMES = ['', 'Blaster', 'Shotgun', 'Super Shotgun', 'Machinegun', 'Chaingun',
         'Grenade Launcher', 'Rocket Launcher', 'HyperBlaster', 'Railgun', 'BFG10K', 'Grenades']


def episode_return(result):
    life = result['first_life']
    damage = life['damage']
    initial_hp = sum(e['health'] for e in result['generated_start']['entities']
                     if e['class'].startswith('monster_'))
    dealt = sum(e['health_damage'] for e in life['outgoing_by_target_class'])
    death = bool(result['death_stop'])
    win = bool(result['goal_stop']) and not death
    return (100 * win - 100 * death + 10 * min(1., dealt / max(1, initial_hp))
            - .1 * damage['received_health_damage'] - damage['self_health_damage']
            - .5 * damage['teammate_health_damage'])


def load_rollouts(paths):
    inputs, rows, episodes, models = {}, [], {}, []
    stream_seeds, native_seeds = set(), None
    for path in paths:
        meta_path = pathlib.Path(str(path) + '.meta.json')
        meta = json.loads(meta_path.read_text())
        if meta['version'] != 'hjepa_onpolicy_weapon_rollout_v1' or meta['split'] != 'train':
            raise ValueError('fresh on-policy TRAIN export required')
        if meta.get('data_sha256') != sha(path):
            raise ValueError('exported rollout changed or lacks data hash')
        for name, digest in meta['sources_sha256'].items():
            if sha(pathlib.Path(name)) != digest:
                raise ValueError('native source or receipt changed: ' + name)
        inputs[str(path.resolve())] = sha(path)
        inputs[str(meta_path.resolve())] = sha(meta_path)
        model_path = pathlib.Path(meta['model'])
        if sha(model_path) != meta['model_sha256']:
            raise ValueError('changed behavior model')
        model = json.loads(model_path.read_text())
        stream = model.get('weapon_sampling_seed')
        if stream is None or stream in stream_seeds or model.get('weapon_option_frames', 0):
            raise ValueError('distinct stochastic frame-wise streams required')
        stream_seeds.add(stream)
        comparable = {k: v for k, v in model.items() if k != 'weapon_sampling_seed'}
        if models and comparable != models[0]:
            raise ValueError('behavior weights differ across streams')
        models.append(comparable)
        current_seeds = set()
        for ep in meta['episodes']:
            key = (stream, ep['seed'])
            if ep['seed'] in current_seeds:
                raise ValueError('duplicate episode')
            current_seeds.add(ep['seed'])
            result = ep['native_result']
            if (result['seed'] != ep['seed'] or not result['capture_valid'] or
                    result['generated_fixture']['split'] != 'train' or
                    not result['generated_start']['confirmed']):
                raise ValueError('invalid native episode binding')
            episodes[key] = dict(expected=ep['rows'], result=result, reward=episode_return(result))
        if native_seeds is not None and current_seeds != native_seeds:
            raise ValueError('incomplete world pairing')
        native_seeds = current_seeds
        exported = [json.loads(line) for line in path.read_text().splitlines()]
        if len(exported) != meta['rows']:
            raise ValueError('export count changed')
        original_steps = {}
        counts = collections.Counter()
        seen = set()
        for row in exported:
            key = (stream, row['seed'])
            if key not in episodes or row['split'] != 'train' or len(row['features']) != 871:
                raise ValueError('invalid rollout row')
            source = row['source']
            if source not in meta['sources_sha256']:
                raise ValueError('unbound step source')
            if source not in original_steps:
                original_steps[source] = {s['index']: s for s in
                    (json.loads(line) for line in pathlib.Path(source).read_text().splitlines())}
            step = original_steps[source][row['step_index']]
            if (step['owner'] != 'provider' or step['observation']['identity'] != row['identity'] or
                    step['action']['weapon'] != NAMES[row['slot']] or
                    not step['server_execution']['matched']):
                raise ValueError('sample differs from actual native choice')
            unique = (source, row['step_index'])
            if unique in seen:
                raise ValueError('duplicate sample')
            seen.add(unique)
            probabilities = row['behavior_probabilities']
            mask, slot = row['availability'], row['slot']
            if (len(mask) != 12 or len(probabilities) != 12 or not mask[slot] or
                    any(not math.isfinite(v) for v in row['features']) or
                    any(p < 0 or not math.isfinite(p) or (p != 0 and not valid)
                        for p, valid in zip(probabilities, mask)) or
                    abs(sum(probabilities)-1) > 1e-10 or
                    abs(math.log(probabilities[slot])-row['behavior_log_probability']) > 1e-10):
                raise ValueError('invalid masked likelihood')
            row['episode_key'] = key
            counts[key] += 1
            rows.append(row)
        if any(counts[(stream, seed)] != episodes[(stream, seed)]['expected'] for seed in current_seeds):
            raise ValueError('missing episode choices')
    if len(paths) < 2 or len(set(map(str, paths))) != len(paths):
        raise ValueError('multiple independent streams required')
    for seed in native_seeds:
        group = [ep for (_, s), ep in episodes.items() if s == seed]
        fixture = group[0]['result']['generated_fixture']
        if any(ep['result']['generated_fixture'] != fixture for ep in group):
            raise ValueError('paired native worlds differ')
        first = {}
        for row in rows:
            if row['seed'] == seed:
                first.setdefault(row['episode_key'], row['features'])
        if any(x != next(iter(first.values())) for x in first.values()):
            raise ValueError('paired initial observations differ')
        for ep in group:
            ep['advantage'] = (ep['reward'] - sum(other['reward'] for other in group if other is not ep)
                               / (len(group)-1)) / 100
    return rows, episodes, models[0], inputs


def model_fingerprint(model):
    import hashlib
    return hashlib.sha256(json.dumps(model,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def verify_prior_inputs(inputs):
    for name, digest in inputs.items():
        path = pathlib.Path(name)
        if sha(path) != digest:
            raise ValueError('prior batch input changed')
        if name.endswith('.meta.json'):
            for source, expected in json.loads(path.read_text())['sources_sha256'].items():
                if sha(pathlib.Path(source)) != expected:
                    raise ValueError('prior native source changed')


def previous_batch(path, parent, new_seeds, credit='episodic'):
    checkpoint = torch.load(path,map_location='cpu',weights_only=False)
    valid_episode=checkpoint['version']==VERSION and checkpoint['algorithm']==ALGORITHM
    valid_rtg=checkpoint['version']==RTG_VERSION and checkpoint['algorithm']==RTG_ALGORITHM
    if not (valid_episode or (credit=='return-to-go' and valid_rtg)):
        raise ValueError('previous batch algorithm mismatch; reconstruct legacy checkpoints first')
    if checkpoint['steps'] != ALGORITHM['max_batch_steps']:
        raise ValueError('previous batch was not completed')
    if set(new_seeds) & set(checkpoint['used_seeds']):
        raise ValueError('TRAIN worlds already used by previous batches')
    model_path = path.parent/'model.json'
    if sha(model_path) != checkpoint['model_sha256'] or json.loads(model_path.read_text()) != parent:
        raise ValueError('behavior model differs from previous batch result')
    state = network(parent['weapon_actor']).state_dict()
    if state.keys() != checkpoint['weapon_actor'].keys() or any(
            not torch.equal(v,checkpoint['weapon_actor'][k].cpu()) for k,v in state.items()):
        raise ValueError('previous optimizer does not bind behavior head')
    verify_prior_inputs(checkpoint['cumulative_inputs'])
    for ancestor in checkpoint['ancestry']:
        if sha(pathlib.Path(ancestor['checkpoint'])) != ancestor['checkpoint_sha256']:
            raise ValueError('ancestor checkpoint changed')
    return checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=pathlib.Path, action='append', required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--steps', type=int, default=4, help='total updates, including resumed updates')
    ap.add_argument('--credit',choices=['episodic','return-to-go'],default='episodic')
    continuation = ap.add_mutually_exclusive_group()
    continuation.add_argument('--resume', type=pathlib.Path, help='resume the same frozen native batch')
    continuation.add_argument('--previous-batch', type=pathlib.Path, help='carry optimizer to a new disjoint native batch')
    a = ap.parse_args()
    if not torch.cuda.is_available() or not 1 <= a.steps <= 4:
        raise ValueError('CUDA required; fixed update budget 1..4')
    rows, episodes, parent, inputs = load_rollouts(a.data)
    version=VERSION if a.credit=='episodic' else RTG_VERSION
    algorithm=ALGORITHM if a.credit=='episodic' else RTG_ALGORITHM
    credit_proof=None;credit_source=None
    if a.credit=='return-to-go':
        import hjepa_weapon_returns
        extra_inputs,credit_proof=hjepa_weapon_returns.build_credit(rows,episodes)
        inputs.update(extra_inputs)
        credit_source=sha(pathlib.Path(hjepa_weapon_returns.__file__))
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'trainer-frozen.py').write_bytes(pathlib.Path(__file__).read_bytes())
    if credit_proof is not None:
        (a.out/'credit-builder-frozen.py').write_bytes(pathlib.Path(hjepa_weapon_returns.__file__).read_bytes())
        save(a.out/'credit-proof.json',credit_proof)
    torch.set_num_threads(2)
    torch.manual_seed(261201)
    torch.cuda.manual_seed_all(261201)
    net = network(parent['weapon_actor']).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    x = torch.tensor([r['features'] for r in rows], device='cuda')
    mask = torch.tensor([r['availability'] for r in rows], device='cuda', dtype=torch.bool)
    slot = torch.tensor([r['slot'] for r in rows], device='cuda')
    old = torch.tensor([r['behavior_probabilities'] for r in rows], device='cuda')
    old_log = torch.tensor([r['behavior_log_probability'] for r in rows], device='cuda')
    adv = torch.tensor([r.get('credit_advantage',episodes[r['episode_key']]['advantage']) for r in rows], device='cuda')
    def log_distribution():
        return net(x).masked_fill(~mask, -1e9).log_softmax(-1)
    with torch.no_grad():
        native_probability_error = float((log_distribution().exp()-old).abs().max())
    if native_probability_error > 1e-4:
        raise ValueError('Go/CUDA behavior probability mismatch')
    start, history = 0, []
    batch_index, total_before, prior_seeds, cumulative_inputs, ancestry = 0, 0, [], dict(inputs), []
    new_seeds = sorted({seed for _,seed in episodes})
    if a.previous_batch:
        previous = previous_batch(a.previous_batch,parent,new_seeds,a.credit)
        optimizer.load_state_dict(previous['optimizer'])
        batch_index = previous['batch_index']+1
        total_before = previous['total_steps']
        prior_seeds = previous['used_seeds']
        cumulative_inputs = {**previous['cumulative_inputs'],**inputs}
        ancestry = previous['ancestry'] + [dict(checkpoint=str(a.previous_batch.resolve()),
                    checkpoint_sha256=sha(a.previous_batch), model_sha256=previous['model_sha256'])]
        torch.set_rng_state(previous['rng'])
        torch.cuda.set_rng_state_all(previous['cuda_rng'])
    if a.resume:
        checkpoint = torch.load(a.resume, map_location='cpu', weights_only=False)
        if (checkpoint['version'] != version or checkpoint['inputs'] != inputs or
                checkpoint.get('credit_source_sha256') != credit_source or
                checkpoint['algorithm'] != algorithm or checkpoint['input_parent'] != model_fingerprint(parent) or
                checkpoint['trainer_sha256'] != sha(pathlib.Path(__file__))):
            raise ValueError('resume contract mismatch')
        net.load_state_dict(checkpoint['weapon_actor'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start, history = checkpoint['steps'], checkpoint['history']
        batch_index, total_before = checkpoint['batch_index'], checkpoint['total_before']
        prior_seeds, cumulative_inputs, ancestry = checkpoint['prior_seeds'], checkpoint['cumulative_inputs'], checkpoint['ancestry']
        verify_prior_inputs(cumulative_inputs)
        torch.set_rng_state(checkpoint['rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        if start >= a.steps:
            raise ValueError('resume must advance updates')
    for update in range(start, a.steps):
        optimizer.zero_grad()
        logp = log_distribution()
        selected = logp.gather(1, slot[:, None]).squeeze(1)
        ratio = (selected-old_log).exp()
        surrogate = torch.minimum(ratio*adv, ratio.clamp(.9,1.1)*adv)
        # Constant 300-frame horizon normalization preserves trajectory-sum
        # credit; dividing each episode by its own length would change objective.
        loss = -surrogate.sum() / (300 * len(episodes))
        if not torch.isfinite(loss):
            raise ValueError('nonfinite objective')
        loss.backward()
        grad = float(nn.utils.clip_grad_norm_(net.parameters(), 1.))
        optimizer.step()
        with torch.no_grad():
            logp = log_distribution()
            kl = float((old * (old.clamp_min(1e-30).log()-logp)).sum(-1).mean())
        if not math.isfinite(kl) or kl > .02:
            raise ValueError('fixed behavior KL budget exceeded; reject update')
        history.append(dict(update=update+1, loss=float(loss.detach()), grad=grad, behavior_kl=kl))
    model = copy.deepcopy(parent)
    model['weapon_actor'] = export(net)
    if {k:v for k,v in model.items() if k!='weapon_actor'} != {k:v for k,v in parent.items() if k!='weapon_actor'}:
        raise ValueError('frozen non-weapon fields changed')
    save(a.out/'model.json', model)
    torch.save(dict(version=version, weapon_actor=net.state_dict(), optimizer=optimizer.state_dict(),
                    steps=a.steps, inputs=inputs, history=history, trainer_sha256=sha(pathlib.Path(__file__)),
                    algorithm=algorithm,credit_source_sha256=credit_source, input_parent=model_fingerprint(parent), model_sha256=sha(a.out/'model.json'),
                    batch_index=batch_index,total_before=total_before,total_steps=total_before+a.steps,
                    prior_seeds=prior_seeds,used_seeds=sorted(set(prior_seeds+new_seeds)),
                    cumulative_inputs=cumulative_inputs,ancestry=ancestry,
                    rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), a.out/'checkpoint.pt')
    cpu = net.cpu().double()
    parity_x = x[:64].cpu().double()
    with torch.no_grad():
        outputs = cpu(parity_x)
        error = float((outputs-serialized_forward(parity_x, model['weapon_actor'])).abs().max())
    if error > 1e-10:
        raise ValueError('serialized head mismatch')
    save(a.out/'parity-vectors.json', dict(features=parity_x.tolist(),outputs=outputs.tolist()))
    report = dict(version=version, promotion=False, rows=len(rows), episodes=len(episodes), steps=a.steps,
                  batch_index=batch_index,total_steps=total_before+a.steps,algorithm=algorithm,ancestry=ancestry,
                  credit_source_sha256=credit_source,
                  native_probability_error=native_probability_error, serialized_error=error,
                  inputs_sha256=inputs, model_sha256=sha(a.out/'model.json'), history=history,
                  episode_returns=[dict(stream=k[0],seed=k[1],reward=v['reward'],advantage=v['advantage'])
                                   for k,v in episodes.items()],
                  scope='Full weapon head only, clipped frame-wise policy gradient with finite-horizon episodic returns '
                        'and independent leave-one-stream-out baseline. Frozen world/motor. No critic or BC loss. '
                        'Guards remain part of native environment; not pure autonomous combat proof.')
    save(a.out/'report.json', report)
    print(json.dumps({k:report[k] for k in ['rows','episodes','history','native_probability_error']},indent=2))


if __name__ == '__main__':
    main()
