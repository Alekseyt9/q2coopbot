"""Freeze an independent geometry confirmation after exploratory control evidence."""
import argparse
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha
from collect_combat_joint_aim_budget_round import historical_plans
from train_combat_bc import torch, training_devices
from combat_joint_aim_policy import JointAim, base_model
from migrate_combat_navigation_cuda import branch
from fork_conditional_weapon_exploration_cuda import equal_tree


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('registry', 'source', 'candidate', 'constant', 'endpoint_decision', 'protocol', 'spec', 'out'):
        ap.add_argument('--'+name.replace('_','-'), type=Path, required=True)
    ap.add_argument('--port', type=int, default=48900)
    a = ap.parse_args(); assert not a.out.exists() and not a.protocol.exists() and not a.spec.exists()
    repo = Path(__file__).resolve().parents[1]; pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve(); h = sha(path)
        assert expected is None or expected == h, str(path)
        pins[str(path)] = h; return path

    decision_seal = read(pin(a.endpoint_decision/'complete.json'))
    decision = read(pin(a.endpoint_decision/'report.json', decision_seal['report_sha256']))
    assert not decision['gate_passed'] and decision['selected_for_expansion'] is None
    assert decision['totals']['initial']['goal'] == 7 and decision['totals']['rules']['goal'] == 4
    index = read(pin(a.registry)); descriptors = [read(pin(a.registry.parent/f)) for f in index['files']]
    assert len(descriptors) == 8 and all('rules' in e['modes'] and 'learned' in e['modes'] and e['timescale'] == 2 for e in descriptors)
    rewards = {sha(pin(repo/e['recipe']['reward_config'])) for e in descriptors}; assert len(rewards) == 1
    models = {}; seals = {}; roots = dict(source=a.source, shared=a.candidate, constant=a.constant)
    for label, root in roots.items():
        seal = read(pin(root/'complete.json')); seals[label] = seal
        for name, key in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):
            pin(root/name, seal[key])
        assert read(root/'report.json')['device'] == 'cuda'
        models[label] = read(root/'weights.json')
        assert not models[label].get('deterministic') and not models[label].get('diagnostic_joint_aim_probe')
    assert seals['shared']['weights_sha256'] == '559ff0ad4297047cdc4a9ad531fa9159668b045b7f41bf8e32a6df975278bea6'
    training_devices(); torch.set_default_device('cuda'); torch.set_num_threads(2)
    references = None
    for label, model in models.items():
        core = base_model(model, allow_shared=True)
        actor, critic, head = branch(core,'actor'), branch(core,'value'), JointAim(model,allow_shared=True)
        std = torch.tensor(model['log_std'],device='cuda')
        architecture = {k:v for k,v in core['attention'].items() if k not in ('actor','value')}
        current = (actor.state_dict(),critic.state_dict(),head.hints.state_dict(),std,architecture)
        if references is None: references = current
        else:
            for old,new in zip(references[:4],current[:4]): assert equal_tree(old,new), 'Unexpected body/critic/predictor/std change: '+label
            assert architecture == references[4]
        if label == 'constant': assert int(torch.count_nonzero(head.head.layers[-1].weight)) == 0
    history = historical_plans(repo/'workspace/artifacts')
    used = {s for _, p in history for t in p['tasks'] for s in t.get('seeds',[])}
    offset = 0
    while True:
        assert all(offset+3 <= e['splits']['confirmation']['count'] for e in descriptors)
        seeds = {e['splits']['confirmation']['start']+offset+i for e in descriptors for i in range(3)}
        if not seeds & used: break
        offset += 3
    assert len(seeds) == 24
    protocol = dict(version='combat_geometry_independent_confirmation_v1', candidate_weights_sha256=seals['shared']['weights_sha256'],
        selection='Exploratory parent choice after the new independent endpoint control gave7/8 vs rules4/8; both PPO arms failed. Prior geometry gate remains false; no criterion is retrospectively changed.',
        split='confirmation', cases_per_variant=24, variants=['source','shared','constant','rules'], native_cases=96, cuda_cases=72,
        maps=sorted({e['map'] for e in descriptors}), max_instances=16, timescale=2, training_performed=False, checkpoint_selection=False,
        training_and_weight_changes_forbidden=True, reward_sha256=next(iter(rewards)),
        criteria=dict(goals_source_margin=4,goals_constant_margin=3,goals_rules_margin=3,deaths_max='rules',received_max='1.1*rules',
            mg_live_fraction_min='source-0.03',large_yaw_max='source+0.02',targetless_attack_max='source+0.03'),
        promotion=False, scope='Independent finite four-map confirmation, not full campaign/coop acceptance. No tuning, no merging previous cohorts, no automatic production promotion.')
    save(a.protocol, protocol)
    variants = [dict(label=label, mode='learned', update=str(root.resolve()), deterministic=False, cuda_replay=True, checked_wall_slide=True) for label,root in roots.items()]
    variants.append(dict(label='rules',mode='rules',cuda_replay=False))
    comparisons = [dict(name=label+'-shared',before=label,after='shared') for label in ('source','constant','rules')]
    save(a.spec, dict(version='registered_combat_model_comparison_v1', registry=str(a.registry.resolve()),split='confirmation',count=3,
        max_instances=16,episodes=sorted(e['id'] for e in descriptors),seed_offset=offset,port=a.port,minimum_free_gib=32,
        ready=[str((a.out/'complete.json').resolve())],variants=variants,comparisons=comparisons,
        historical_plans=[str(p) for p,_ in history],unavailable_historical_references=historical_plans.missing_references,
        confirmation_protocol=str(a.protocol.resolve()),confirmation_protocol_sha256=sha(a.protocol),training_performed=False,promotion=False))
    a.out.mkdir(parents=True)
    for path,h in pins.items(): assert sha(path) == h, path
    report = dict(state='complete',version='combat_geometry_independent_confirmation_preparation_v1',device='cuda',
        body_critic_predictor_std_exact_across_models=True,constant_head_state_independent=True,neural_weight_changes=False,
        weights={k:s['weights_sha256'] for k,s in seals.items()},split='confirmation',seed_offset=offset,seeds=sorted(seeds),
        protocol_sha256=sha(a.protocol),spec_sha256=sha(a.spec),inputs=pins,training_performed=False,promotion=False,script_sha256=sha(__file__))
    save(a.out/'report.json',report); save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json'),protocol_sha256=sha(a.protocol),spec_sha256=sha(a.spec)))
    print(dict(state='complete',native_cases=96,cuda_cases=72,split='confirmation',seed_offset=offset,body_exact=True))


if __name__ == '__main__': main()
