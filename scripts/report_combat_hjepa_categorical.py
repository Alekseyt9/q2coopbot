"""Native learning versus untrained categorical-exploration controls."""
import argparse
import collections
import copy
import json
import pathlib
from train_combat_hjepa import save, sha
from report_combat_hjepa_weapon_stochastic import load, summarize, bootstrap


def jsonl_rows(path):
    with path.open(encoding='utf-8') as stream:
        yield from map(json.loads,stream)


def native_cycles(root, role):
    totals = dict(hold_decisions=0,proposed_releases=0,applied_releases=0,held_grenade_hp=0,held_grenade_events=0)
    sources = {}
    for report_path in (root/role).glob('case-*/report.json'):
        report = json.loads(report_path.read_text())
        for result in report['results']:
            dataset = report_path.parent/f'worker-{result["worker"]}-episode-{result["episode"]}'/'dataset'
            life = result['first_life'];actor = life['bot_entity']
            steps_path, outcomes_path = dataset/'steps.jsonl', dataset/'server_outcomes.jsonl'
            sources[str(steps_path.resolve())],sources[str(outcomes_path.resolve())] = sha(steps_path),sha(outcomes_path)
            for step in jsonl_rows(steps_path):
                o = step['observation'];frame = o['identity']['frame'];weapon = o['weapon'].lower()
                if (step['owner'] != 'provider' or not life['start_frame'] <= frame < life['end_frame']
                        or not step.get('server_execution',{}).get('window_exclusive')
                        or o['gun_frame'] != 11 or not ('v_handgr' in weapon or weapon == 'grenades')):
                    continue
                totals['hold_decisions'] += 1
                totals['proposed_releases'] += int(not step['action']['attack'])
                totals['applied_releases'] += int(not step['applied_action']['attack'])
            self_hp = 0
            for outcome in jsonl_rows(outcomes_path):
                for e in outcome['events']:
                    if e['attacker'] == actor and e['target'] == actor and life['start_frame'] <= e['frame'] <= life['end_frame']:
                        loss = max(0,min(max(0,e['health_before']),e['take'])); self_hp += loss
                        if e['mod'] == 24:
                            totals['held_grenade_hp'] += loss;totals['held_grenade_events'] += 1
            if self_hp != life['damage']['self_health_damage']:
                raise ValueError('self-damage native ledger mismatch')
    return totals, sources


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=pathlib.Path,required=True)
    a = ap.parse_args();a.root = a.root.resolve()
    data,models,fixtures = load(a.root/'native-validation',worlds=16,split='validation')
    controls,_,control_fixtures = load(a.root/'rules-validation',worlds=16,split='validation')
    expected = {f'{role}-s{s}' for role in ['cold','hot','trained'] for s in range(2)}|{'source-mean','trained-mean','live-u1'}
    if set(data) != expected or set(controls) != {'rules'} or fixtures != control_fixtures:
        raise ValueError('incomplete matched evaluation')
    data.update(controls)
    for stream in range(2):
        for role in ['cold','hot','trained']:
            model = models[f'{role}-s{stream}']
            if model['primitive_sampling']['seed'] != 262500+stream or model['primitive_sampling']['categorical_temperature'] != (.25 if role=='cold' else 1.):
                raise ValueError('uncoupled or changed evaluation distribution')
        cold = copy.deepcopy(models[f'cold-s{stream}']);cold['primitive_sampling']['categorical_temperature'] = 1.
        if cold != models[f'hot-s{stream}']:
            raise ValueError('untrained exploration changes more than temperature')
        trained = copy.deepcopy(models[f'trained-s{stream}'])
        trained['levels'][0]['encoder'] = cold['levels'][0]['encoder']
        trained['actor'] = cold['actor'];trained['primitive_sampling']['log_std'] = cold['primitive_sampling']['log_std']
        if trained != cold:
            raise ValueError('nonprimitive component changed')
    for role,source in [('source-mean','cold-s0'),('trained-mean','trained-s0')]:
        mean = copy.deepcopy(models[source]);mean.pop('primitive_sampling')
        if mean != models[role]:
            raise ValueError('mean evaluation weights mismatch')
    pooled = {role:summarize([r for s in range(2) for r in data[f'{role}-s{s}'].values()]) for role in ['cold','hot','trained']}
    pooled.update({role:summarize(list(data[role].values())) for role in ['source-mean','trained-mean','live-u1','rules']})
    cycles,sources = {},{}
    for role in expected-{'live-u1'}:
        cycles[role],bound = native_cycles(a.root/'native-validation',role);sources.update(bound)
    for role in ['cold','hot','trained']:
        cycles[role] = {k:sum(cycles[f'{role}-s{s}'][k] for s in range(2)) for k in cycles[f'{role}-s0']}
    old,new = pooled['hot']['totals'],pooled['trained']['totals']
    before,after = pooled['source-mean']['totals'],pooled['trained-mean']['totals']
    criteria = dict(sampled=dict(win_gain=new['win'] >= old['win']+4,deaths_not_increased=new['death'] <= old['death'],
        received_not_increased=new['received_hp'] <= old['received_hp'],self_not_increased=new['self_hp'] <= old['self_hp']),
        primitive_mean=dict(win_gain=after['win'] >= before['win']+2,deaths_not_increased=after['death'] <= before['death'],
        received_not_increased=after['received_hp'] <= before['received_hp'],self_not_increased=after['self_hp'] <= before['self_hp']))
    comparisons = {}
    for reference in ['hot','cold']:
        clusters = collections.defaultdict(list)
        for seed in fixtures:
            clusters[data['trained-s0'][seed]['map']].append({k:sum(data[f'trained-s{s}'][seed][k]-data[f'{reference}-s{s}'][seed][k] for s in range(2))/2 for k in ['win','death','received_hp','self_hp','score']})
        comparisons['trained-minus-'+reference] = {k:bootstrap([sum(r[k] for r in group)/len(group) for group in clusters.values()]) for k in ['win','death','received_hp','self_hp','score']}
    report = dict(version='hjepa_categorical_curriculum_screen_v1',promotion=False,roles=pooled,cycles=cycles,criteria=criteria,
        pilot_gate_passed=all(all(v.values()) for v in criteria.values()),paired_map_cluster_intervals=comparisons,
        cycle_source_sha256=sources,protocol_sha256=sha(a.root/'protocol.json'),
        scope='Three fresh on-policy batches,12natural-gradient updates,48TRAINworlds/192games. Evaluation160actualgames on16fresh validation446worlds,eight knownmaps; sampled roles have32games/two coupled primitive streams, controls16each. Cold->hot changes global fire AND pose temperature without learning; hot->trained changes learned primitive weights/log_std only. Weapon head frozen. Native held-grenade health losses capped at available HP; survival-time/weapon-use differences confound conditional cycle counts. Rules useslegacy_v1; learnedstock_weapons_fire_v1. No global unseen-map, independent replication or promotion claim; reserved442unplayed.')
    save(a.root/'native-screen.json',report)
    print(json.dumps(dict(roles=pooled,cycles={k:cycles[k] for k in ['cold','hot','trained','source-mean','trained-mean']},criteria=criteria,pilot_gate_passed=report['pilot_gate_passed']),indent=2))


if __name__ == '__main__':
    main()
