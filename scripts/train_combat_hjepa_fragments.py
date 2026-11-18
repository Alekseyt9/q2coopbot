"""CUDA joint primitive/weapon update on source-bound short native fragments.

Only native-terminal same-stream Monte Carlo targets are accepted. Repeated
branches are grouped by checkpoint family; diagnostics never enter this loader.
"""
import argparse
import collections
import copy
import json
import math
import pathlib

import torch
from torch import nn
from torch.nn import functional as F

from train_combat_hjepa import export, save, sha
from train_combat_hjepa_actor_ppo import Actor, network, log_probability, distribution_kl, fingerprint

VERSION = 'hjepa_short_fragment_joint_ppo_v1'
ALGORITHM = dict(learning_rate=1e-4, clip=.1, max_mean_kl=.02, gradient_clip=1., advantage_scale=100., credit='same_root_other_stream_native_terminal_mc_v1')


def distribution_without_sampling_seeds(model):
    spec = copy.deepcopy(model)
    primitive = spec.get('primitive_sampling')
    if not primitive or any(type(seed) is not int or not 0 <= seed <= 9223372036854775807
                            for seed in (primitive.get('seed'),spec.get('weapon_sampling_seed'))):
        raise ValueError('explicit valid primitive and weapon sampling seeds required')
    primitive['seed'], spec['weapon_sampling_seed'] = 0, 0
    return spec


def load(paths, seed_equivalent=False):
    inputs, families, rows, parent = {}, {}, [], None
    family_by_condition_root, worlds = {}, set()
    scopes = set()
    parent_distribution, parent_distribution_key, model_seeds, family_models = None, None, {}, {}
    for path in paths:
        target = json.loads(path.read_text())
        if target['version'] not in ('hjepa_fragment_mc_targets_v1','hjepa_fragment_mc_targets_v2','hjepa_fragment_mc_targets_v3','hjepa_fragment_mc_targets_v4') or not target['optimizer_ready'] or target['bootstrap_proof'] != 'native_terminal_in_same_stream_continuation':
            raise ValueError('verified native terminal targets required')
        for name, digest in target['sources_sha256'].items():
            if sha(pathlib.Path(name)) != digest:
                raise ValueError('changed target source: '+name)
            inputs[name] = digest
        inputs[str(path.resolve())] = sha(path)
        fragment = json.loads(pathlib.Path(target['prefix']).read_text())
        if fragment['version'] not in ('hjepa_native_fragment_v2','hjepa_native_fragment_v3','hjepa_native_fragment_v4','hjepa_native_fragment_v5') or fragment['split'] != 'train' or not fragment['training_source_verified'] or not fragment['training_plan']:
            raise ValueError('typed verified TRAIN fragment required')
        required = {target['prefix'], target['tail'], fragment['behavior_model'], fragment['training_plan']}
        if not required.issubset(target['sources_sha256']) or any(
                target['sources_sha256'].get(name) != digest for name, digest in fragment['sources_sha256'].items()):
            raise ValueError('target must bind prefix, tail and all fragment sources')
        if (target['model_version'] != fragment['model_version'] or target['bootstrap_value'] != 0
                or target['terminal_reason'] not in ('native_death', 'native_source_roster_clear', 'native_objective_clear')
                or target['terminal_step'] < len(fragment['transitions'])-1):
            raise ValueError('native terminal/model binding differs')
        if target['terminal_reason'] == 'native_objective_clear':
            state, identity = target.get('terminal_native_objective', {}), target.get('terminal_next_identity', {})
            if (state.get('alive') != 0 or state.get('pending') != 0 or not identity
                    or any(state.get(a) != identity.get(b) for a,b in [('map','map'),('spawncount','spawncount'),('frame','frame')])):
                raise ValueError('native objective zero state must match terminal next identity')
        model_path = pathlib.Path(fragment['behavior_model'])
        model = json.loads(model_path.read_text())
        if ('hjepa:'+sha(model_path) != fragment['model_version'] or model['weapon_mode'] != 'learned_masked_v1'
                or model['weapon_head'] != 'combat_masked_weapon_v1' or model.get('weapon_sampling_seed') is None
                or model['planner'] != 'bc' or model.get('motor') or model.get('weapon_option_frames', 0)):
            raise ValueError('unchanged full joint behavior actor required')
        distribution = distribution_without_sampling_seeds(model) if seed_equivalent else model
        if parent is not None and distribution != parent_distribution:
            raise ValueError('different behavior weights')
        if parent is None:
            parent, parent_distribution = model, distribution
            if seed_equivalent:parent_distribution_key = fingerprint(distribution)
        if seed_equivalent:
            model_seeds[fragment['model_version']] = dict(primitive_seed=model['primitive_sampling']['seed'],weapon_seed=model['weapon_sampling_seed'])
        family, stream = fragment['checkpoint_family'], fragment['sampling_checkpoint']['sampling_stream']
        checkpoint = fragment['sampling_checkpoint']
        scope, version = checkpoint.get('sampling_scope', ''), checkpoint.get('version')
        if not 0 <= stream <= 2147483647 or version == 'hjepa_sampling_checkpoint_v1' and stream != 0 or not ((version in ('hjepa_sampling_checkpoint_v1','hjepa_sampling_checkpoint_v2') and not scope)
                or version == 'hjepa_sampling_checkpoint_v3' and scope == 'fire_only_v1'
                or version == 'hjepa_sampling_checkpoint_v4' and scope == 'root_fire_only_v1'
                or version == 'hjepa_sampling_checkpoint_v5' and scope == 'root_joint_only_v1'):
            raise ValueError('sampling checkpoint scope/version mismatch')
        if target.get('sampling_scope', '') != scope:
            raise ValueError('target sampling scope differs')
        expected = {'':('hjepa_native_fragment_v2','hjepa_fragment_mc_targets_v1'),
                    'fire_only_v1':('hjepa_native_fragment_v3','hjepa_fragment_mc_targets_v2'),
                    'root_fire_only_v1':('hjepa_native_fragment_v4','hjepa_fragment_mc_targets_v3'),
                    'root_joint_only_v1':('hjepa_native_fragment_v5','hjepa_fragment_mc_targets_v4')}[scope]
        if (fragment['version'],target['version']) != expected:
            raise ValueError('conditional sampling requires new fragment and target versions')
        scopes.add(scope)
        if len(scopes) != 1:
            raise ValueError('mixed joint/fire-only sampling scopes')
        source_report = pathlib.Path(fragment['source_directory'])/'report.json'
        if str(source_report) not in fragment['sources_sha256']:
            raise ValueError('native source report must be hash-bound')
        source = json.loads(source_report.read_text())
        condition = source.get('generated_fixture', {})
        origin = fragment['sampling_checkpoint'].get('source_identity', {})
        if (condition.get('split') != 'train' or not source.get('generated_start', {}).get('confirmed')
                or origin.get('map') != condition.get('map') or origin.get('frame', 0) < 1):
            raise ValueError('native TRAIN condition/logical root missing')
        condition_root = fingerprint(dict(condition=condition, frame=origin['frame'], model=fragment['model_version']))
        if seed_equivalent:
            condition_root = fingerprint(dict(condition=condition,frame=origin['frame'],distribution=parent_distribution_key,seeds=model_seeds[fragment['model_version']]))
        if family_by_condition_root.setdefault(condition_root, family) != family:
            raise ValueError('same TRAIN condition/root recaptured under another checkpoint family')
        if family_models.setdefault(family,fragment['model_version']) != fragment['model_version']:
            raise ValueError('one checkpoint family cannot change behavior model sampling seeds')
        worlds.add(fingerprint(condition))
        bank = families.setdefault(family, {})
        if stream in bank:
            raise ValueError('duplicate family/stream, including repeated mechanical runs')
        transitions = fragment['transitions']
        labels = target['targets']
        if not 1 <= len(transitions) <= 100:
            raise ValueError('short TRAIN prefix must contain 1..100 transitions; long tails supply credit only')
        if len(transitions) != len(labels) or target['sampling_stream'] != stream or target['checkpoint_family'] != family:
            raise ValueError('target fragment binding/length')
        bank[stream] = {r['index']: t['return_to_go'] for r,t in zip(transitions, labels) if r['index'] == t['index']}
        if len(bank[stream]) != len(transitions):
            raise ValueError('target indexes differ')
        if not all(math.isfinite(value) for value in bank[stream].values()):
            raise ValueError('nonfinite terminal return')
        for row in transitions:
            if seed_equivalent and (fragment.get('base_sampling_seed') != model['primitive_sampling']['seed']
                    or fragment['sampling_checkpoint'].get('model') != fragment['model_version']
                    or row.get('sample',{}).get('sampling_seed') != model['primitive_sampling']['seed']):
                raise ValueError('behavior sampling seed binding differs')
            if row['version'] != 'hjepa_native_fragment_transition_v2' or len(row['features']) != 871 or len(row['actor_raw']) != 8:
                raise ValueError('fragment feature/logit version')
            if not all(math.isfinite(v) for v in row['features']+row['actor_raw']):
                raise ValueError('nonfinite fragment inputs')
            rows.append((family, stream, row))
    if not rows:
        raise ValueError('empty TRAIN fragments')
    packed = []
    if scopes == {'root_joint_only_v1'}:
        validate_root_joint_records(rows)
    if scopes == {'root_fire_only_v1'}:
        cfg = parent['primitive_sampling']
        temperature = cfg.get('fire_temperature')
        packed = enumerated_fire_roots(rows,families,cfg['categorical_temperature'] if temperature is None else temperature)
    else:
        for family, stream, row in rows:
            peers = [bank[row['index']] for key,bank in families[family].items() if key != stream and row['index'] in bank]
            if not peers:
                raise ValueError('at least two paired streams per checkpoint family required')
            advantage = (families[family][stream][row['index']]-sum(peers)/len(peers))/100.
            entry = dict(row, checkpoint_family=family)
            if scopes == {'root_joint_only_v1'}:
                entry['sampling_stream'] = stream
            packed.append((entry, advantage))
    admission = dict(rows=len(rows), checkpoint_families=len(families), training_worlds=len(worlds), streams={key: sorted(bank) for key,bank in families.items()})
    if scopes != {''}:
        admission['sampling_scope'] = next(iter(scopes))
    if seed_equivalent:
        admission['sampling_seed_equivalence'] = 'identical_distribution_except_two_seed_fields_v1'
        admission['distribution_sha256'] = parent_distribution_key
        admission['behavior_model_versions'] = model_seeds
    return parent, packed, inputs, admission


def validate_root_joint_records(records, stream_count=None):
    groups = collections.defaultdict(list)
    for family, stream, row in records:
        groups[family].append((stream, row))
    for family, group in groups.items():
        streams = sorted(stream for stream, row in group)
        if (not 2 <= len(group) <= 16 or streams != list(range(1,len(group)+1))
                or stream_count is not None and len(group) != stream_count):
            raise ValueError('complete predeclared consecutive root-joint streams required')
        root = group[0][1]
        for stream, row in group:
            if (row['index'] != 0 or row['features'] != root['features']
                    or row['actor_raw'] != root['actor_raw']
                    or row['logical_sampling_identity'] != root['logical_sampling_identity']):
                raise ValueError('root-joint credit requires common inputs and only the first decision')


def enumerated_fire_roots(records, returns, temperature):
    groups = collections.defaultdict(list)
    for family,stream,row in records:groups[family].append((stream,row))
    packed = []
    for family, entries in groups.items():
        if len(entries) != 2 or len({stream for stream,row in entries}) != 2 or {row['sample']['attack'] for stream,row in entries} != {False,True}:
            raise ValueError('root action enumeration requires exactly two complete opposite-action streams')
        validate_fire_roots([dict(row,checkpoint_family=family) for stream,row in entries])
        if entries[0][1]['actor_raw'] != entries[1][1]['actor_raw']:
            raise ValueError('root enumeration actor logits differ')
        x = entries[0][1]['actor_raw'][4]/temperature
        probability = 1/(1+math.exp(-x)) if x >= 0 else math.exp(x)/(1+math.exp(x))
        if not 0 < probability < 1:
            raise ValueError('root enumeration requires two positive behavior action masses')
        q = {row['sample']['attack']:returns[family][stream][0] for stream,row in entries}
        baseline = probability*q[True]+(1-probability)*q[False]
        for stream,row in entries:
            mass = probability if row['sample']['attack'] else 1-probability
            packed.append((dict(row,checkpoint_family=family,enumeration_mass=mass),(q[row['sample']['attack']]-baseline)/100.))
    return packed


def validate_fire_roots(rows):
    """The conditional objective requires identical non-fire root proposals."""
    common = {}
    for row in rows:
        sample = row['sample']
        key = (row['features'], sample['latent'], sample['vertical'], sample.get('weapon',0))
        if row['index'] != 0 or row.get('logical_sampling_identity',{}).get('frame',0) < 1:
            raise ValueError('fire conditional fitting requires one decision at each saved root')
        if common.setdefault(row['checkpoint_family'],key) != key:
            raise ValueError('fire-only roots changed non-fire inputs/draws')


def held_grenade_phase(row):
    features = row['features']
    if len(features) != 871:
        raise ValueError('observed phase weighting requires H-JEPA context871')
    return features[845+11] > .5 and abs(features[5]-11/30) < 1e-6


def family_weights(rows, phase_balanced=False):
    counts = collections.Counter(row['checkpoint_family'] for row in rows)
    if phase_balanced:
        groups = collections.Counter((row['checkpoint_family'], held_grenade_phase(row)) for row in rows)
        strata = collections.Counter(family for family,phase in groups)
        return [len(rows)/(len(counts)*strata[row['checkpoint_family']]*groups[(row['checkpoint_family'],held_grenade_phase(row))]) for row in rows]
    return [len(rows)/(len(counts)*counts[row['checkpoint_family']]) for row in rows]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--targets', type=pathlib.Path, nargs='+', required=True)
    p.add_argument('--out', type=pathlib.Path, required=True)
    p.add_argument('--updates', type=int, default=1)
    p.add_argument('--resume', type=pathlib.Path)
    p.add_argument('--family-balanced', action='store_true', help='equal total surrogate/KL weight per checkpoint family')
    p.add_argument('--phase-balanced', action='store_true', help='equal mass for present observed hold11/other strata within each family; changes the state-weighted objective')
    p.add_argument('--fire-conditional', action='store_true', help='root-only conditional fire surrogate; requires verified common non-fire draws')
    p.add_argument('--fire-head-only', action='store_true', help='fit only the last actor layer fire row; experimental training ablation, full runtime actor remains learned')
    p.add_argument('--root-counterfactual', action='store_true', help='two sampled root actions, common future RNG, exact frozen Bernoulli mass weighting')
    p.add_argument('--root-joint-streams', type=int, default=0, help='predeclared IID joint root streams1..N, common future RNG; require all N terminal targets')
    p.add_argument('--seed-equivalent-models', action='store_true', help='allow behavior copies differing only in primitive/weapon sampling seeds')
    p.add_argument('--reference-model', type=pathlib.Path, help='explicit frozen output sampling/reference spec when admitting seed-equivalent copies')
    a = p.parse_args()
    if not 1 <= a.updates <= 4 or a.out.exists() or not torch.cuda.is_available():
        raise ValueError('fresh output, CUDA and1..4 bounded updates required')
    if a.seed_equivalent_models != bool(a.reference_model):
        raise ValueError('seed-equivalent models require an explicit reference-model and matching flag')
    parent, packed, inputs, admission = load(a.targets,a.seed_equivalent_models)
    if a.reference_model:
        reference = json.loads(a.reference_model.read_text())
        if distribution_without_sampling_seeds(reference) != distribution_without_sampling_seeds(parent):
            raise ValueError('reference model changes behavior distribution weights')
        parent = reference
        inputs[str(a.reference_model.resolve())] = sha(a.reference_model)
    if (a.fire_conditional != (admission.get('sampling_scope') == 'fire_only_v1')
            or a.root_counterfactual != (admission.get('sampling_scope') == 'root_fire_only_v1')
            or bool(a.root_joint_streams) != (admission.get('sampling_scope') == 'root_joint_only_v1')
            or a.root_joint_streams != 0 and not 2 <= a.root_joint_streams <= 16
            or a.fire_head_only and not (a.fire_conditional or a.root_counterfactual)):
        raise ValueError('sampling scope requires matching explicit objective; head-only requires conditional fire')
    if a.root_joint_streams:
        validate_root_joint_records([(r['checkpoint_family'], r['sampling_stream'], r) for r, _ in packed], a.root_joint_streams)
    if a.phase_balanced and (not a.family_balanced or parent['feature_version'] != 'hjepa_context_features_v1'):
        raise ValueError('phase-balanced requires family-balanced and context871')
    algorithm = dict(ALGORITHM, family_weighting='equal_family_v1') if a.family_balanced else ALGORITHM
    if a.phase_balanced:
        algorithm = dict(algorithm,phase_weighting='observed_hold11_within_family_equal_mass_v1')
    if a.fire_conditional:
        algorithm = dict(algorithm,policy_objective='fire_conditional_common_random_root_v1')
    if a.root_counterfactual:
        algorithm = dict(algorithm,policy_objective='clipped_root_fire_action_enumeration_v1',credit='paired_root_actions_shared_future_native_mc_v1',action_mass='frozen_behavior_bernoulli_v1')
    if a.root_joint_streams:
        algorithm = dict(algorithm,policy_objective='clipped_joint_root_loo_v1',credit='iid_joint_root_common_future_native_mc_v1',root_stream_count=a.root_joint_streams)
    if a.fire_head_only:
        algorithm = dict(algorithm,parameter_update='last_actor_fire_row_v1',learning_rate=1e-2)
    if a.seed_equivalent_models:
        algorithm = dict(algorithm,sampling_seed_equivalence='identical_distribution_except_two_seed_fields_v1')
    torch.manual_seed(1729)
    model = Actor(parent).cuda()
    weapon = network(parent['weapon_actor']).cuda()
    parameters = list(model.parameters())+list(weapon.parameters())
    if a.fire_head_only:
        for parameter in parameters:parameter.requires_grad_(False)
        last = model.actor[-1]
        if not isinstance(last,nn.Linear) or last.out_features != 8:
            raise ValueError('head-only requires an eight-output linear actor layer')
        parameters = [last.weight,last.bias]
        for parameter in parameters:
            parameter.requires_grad_(True)
            mask_row = torch.zeros_like(parameter);mask_row[4] = 1
            parameter.register_hook(lambda gradient,mask_row=mask_row:gradient*mask_row)
    optimizer = torch.optim.Adam(parameters, lr=algorithm['learning_rate'])
    tensor = lambda values, dtype=torch.float64: torch.tensor(values, dtype=dtype, device='cuda')
    rows = [r for r,_ in packed]
    if a.fire_conditional or a.root_counterfactual:validate_fire_roots(rows)
    weights = tensor(family_weights(rows, a.phase_balanced) if a.family_balanced else [1.]*len(rows))
    loss_weights = weights*tensor([2*r['enumeration_mass'] for r in rows]) if a.root_counterfactual else weights
    if a.phase_balanced:
        admission = dict(admission,observed_hold11_rows=sum(held_grenade_phase(row) for row in rows))
    x, old_raw = tensor([r['features'] for r in rows]), tensor([r['actor_raw'] for r in rows])
    latent = tensor([r['sample']['latent'] for r in rows])
    attack = tensor([r['sample']['attack'] for r in rows], torch.bool)
    pose = tensor([r['sample']['vertical'] for r in rows], torch.long)
    chosen = tensor([r['sample'].get('weapon', 0) for r in rows], torch.long)
    mask = x[:, 833:845] > .5
    old_lp = tensor([r['sample']['log_probability'] for r in rows])
    old_fire_lp = F.logsigmoid(torch.where(attack,old_raw[:,4],-old_raw[:,4])/model.fire_temperature)
    advantage = tensor([v for _,v in packed])
    old_std = model.log_std.detach().clone()
    with torch.no_grad():
        old_weapon = F.log_softmax(weapon(x).masked_fill(~mask, -torch.inf), -1)

    def current():
        raw = model(x)
        wp = F.log_softmax(weapon(x).masked_fill(~mask, -torch.inf), -1)
        lp = log_probability(raw, model.log_std, latent, attack, pose, model.temperature, model.fire_temperature)+wp.gather(1, chosen[:,None])[:,0]
        score_lp = F.logsigmoid(torch.where(attack,raw[:,4],-raw[:,4])/model.fire_temperature) if a.fire_conditional or a.root_counterfactual else lp
        ratio = (score_lp-(old_fire_lp if a.fire_conditional or a.root_counterfactual else old_lp)).exp()
        surrogate = (loss_weights*torch.minimum(ratio*advantage, ratio.clamp(.9,1.1)*advantage)).mean()
        wk = torch.where(mask, old_weapon.exp()*(old_weapon-wp), 0.).sum(-1)
        kl = distribution_kl(old_raw, old_std, raw, model.log_std, model.temperature, model.fire_temperature)+wk
        return surrogate, (weights*kl).mean(), kl.max(), lp

    with torch.no_grad():
        raw_error = float((model(x)-old_raw).abs().max())
        lp_error = float((current()[3]-old_lp).abs().max())
    if raw_error > 1e-8 or lp_error > 1e-7:
        raise ValueError(f'Go/CUDA initial joint density mismatch: {raw_error}, {lp_error}')
    binding = fingerprint(inputs)
    history = []
    if a.resume:
        ck = torch.load(a.resume, map_location='cuda', weights_only=False)
        if ck['version'] != VERSION or ck['algorithm'] != algorithm or ck['binding'] != binding or ck['parent'] != fingerprint(parent):
            raise ValueError('resume algorithm/input/model mismatch')
        model.load_state_dict(ck['actor']);weapon.load_state_dict(ck['weapon']);optimizer.load_state_dict(ck['optimizer']);history=ck['history']
        if len(history) >= a.updates:
            raise ValueError('resume must advance same batch')
    for index in range(len(history),a.updates):
        before = copy.deepcopy(model.state_dict()), copy.deepcopy(weapon.state_dict()), copy.deepcopy(optimizer.state_dict())
        optimizer.zero_grad(set_to_none=True)
        prior = float(current()[0].detach())
        (-current()[0]).backward()
        grad = float(nn.utils.clip_grad_norm_(parameters,1.))
        optimizer.step()
        with torch.no_grad():model.log_std.clamp_(-10,1)
        proposal = copy.deepcopy(model.state_dict()), copy.deepcopy(weapon.state_dict())
        accepted = False
        with torch.no_grad():
            for power in range(13):
                scale = 2.**(-power)
                model.load_state_dict({k:v+scale*(proposal[0][k]-v) for k,v in before[0].items()})
                weapon.load_state_dict({k:v+scale*(proposal[1][k]-v) for k,v in before[1].items()})
                gain, mean_kl, max_kl, _ = current()
                metrics = [float(gain),float(mean_kl),float(max_kl)]
                if all(math.isfinite(v) for v in metrics) and metrics[1] <= .02 and metrics[0] >= prior-1e-12:
                    accepted = True;break
        if not accepted:
            model.load_state_dict(before[0]);weapon.load_state_dict(before[1]);optimizer.load_state_dict(before[2]);break
        history.append(dict(update=index+1,surrogate=metrics[0],mean_kl=metrics[1],max_kl=metrics[2],step_scale=scale,gradient_norm=grad))
    candidate = copy.deepcopy(parent)
    candidate['levels'][0]['encoder'], candidate['actor'], candidate['weapon_actor'] = export(model.encoder), export(model.actor), export(weapon)
    candidate['primitive_sampling']['log_std'] = model.log_std.detach().cpu().tolist()
    a.out.mkdir(parents=True)
    save(a.out/'model.json', candidate)
    torch.save(dict(version=VERSION,algorithm=algorithm,binding=binding,parent=fingerprint(parent),actor=model.state_dict(),weapon=weapon.state_dict(),optimizer=optimizer.state_dict(),history=history),a.out/'checkpoint.pt')
    with torch.no_grad():
        save(a.out/'parity-actor.json',dict(features=x[:8].cpu().tolist(),outputs=model(x[:8]).cpu().tolist()))
        save(a.out/'parity-weapon.json',dict(features=x[:8].cpu().tolist(),outputs=weapon(x[:8]).cpu().tolist()))
    scope = 'Offline conditional fire-root pilot with full distribution KL; runtime ownership follows the explicit model scope, no independent combat quality evidence' if a.fire_conditional or a.root_counterfactual else 'Offline joint primitive/weapon fragment pilot; no independent combat quality evidence'
    report = dict(version=VERSION,algorithm=algorithm,admission=admission,initial_parity=dict(raw=raw_error,joint_log_probability=lp_error),history=history,binding=binding,inputs_sha256=inputs,model_sha256=sha(a.out/'model.json'),device=torch.cuda.get_device_name(),promotion=False,scope=scope)
    save(a.out/'report.json',report)
    print(json.dumps({k:report[k] for k in ['admission','initial_parity','history','device','promotion']},indent=2))


if __name__ == '__main__':
    main()
