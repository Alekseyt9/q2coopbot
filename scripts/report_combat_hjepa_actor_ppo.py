"""Fresh-world first primitive PPO screen, with paired map-cluster intervals."""
import argparse
import collections
import copy
import json
import pathlib
from report_combat_hjepa_weapon_stochastic import load, summarize, bootstrap
from train_combat_hjepa import save


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--matrix', type=pathlib.Path, required=True)
    ap.add_argument('--rules-matrix', type=pathlib.Path)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    a = ap.parse_args()
    data, models, fixtures = load(a.matrix, worlds=16, split='validation')
    if a.rules_matrix:
        controls, _, control_fixtures = load(a.rules_matrix, worlds=16, split='validation')
        if set(controls) != {'rules'} or 'rules' in data or control_fixtures != fixtures:
            raise ValueError('different rules cohort')
        data.update(controls)
    if set(data) != {'parent', 'candidate', 'parent-sampled', 'candidate-sampled', 'live-u1', 'rules'}:
        raise ValueError('incomplete PPO comparison')
    for role in ('parent', 'candidate'):
        sampled = copy.deepcopy(models[role+'-sampled'])
        sampled.pop('primitive_sampling')
        if sampled != models[role]:
            raise ValueError('sampled/mean behavior weights differ')
    before, after = models['parent-sampled'], copy.deepcopy(models['candidate-sampled'])
    after['levels'][0]['encoder'] = before['levels'][0]['encoder']
    after['actor'] = before['actor']
    after['primitive_sampling']['log_std'] = before['primitive_sampling']['log_std']
    if before != after:
        raise ValueError('nonprimitive model changed')
    roles = {name: summarize(list(values.values())) for name, values in data.items()}
    keys = ['win', 'death', 'received_hp', 'self_hp', 'score']
    comparisons = {}
    for candidate, reference in [('candidate','parent'),('candidate-sampled','parent-sampled'),
                                 ('candidate','live-u1'),('candidate','rules')]:
        clusters = collections.defaultdict(list)
        for seed in fixtures:
            clusters[data[candidate][seed]['map']].append({k:data[candidate][seed][k]-data[reference][seed][k] for k in keys})
        comparisons[candidate+'-minus-'+reference] = {
            k:bootstrap([sum(r[k] for r in group)/len(group) for group in clusters.values()]) for k in keys}
    criteria = {}
    for suffix in ('', '-sampled'):
        old, new = roles['parent'+suffix]['totals'], roles['candidate'+suffix]['totals']
        criteria['sampled' if suffix else 'primitive_mean'] = dict(win_gain=new['win'] >= old['win']+2,
            deaths_not_increased=new['death'] <= old['death'],received_not_increased=new['received_hp'] <= old['received_hp'],
            self_not_increased=new['self_hp'] <= old['self_hp'])
    report = dict(version='hjepa_primitive_ppo_native_screen_v1', promotion=False, roles=roles,
                  paired_map_cluster_intervals=comparisons, criteria=criteria,
                  pilot_gate_passed=all(all(v.values()) for v in criteria.values()),
                  scope='16 fresh validation444 worlds, 96 actual games, eight known maps. One coupled primitive/weapon stream per sampled policy; primitive-mean policy still has frozen stochastic weapon selection. Learned policies use stock_weapons_fire_v1; rules uses its supported legacy_v1 dispatch, so that control includes execution-guard differences. No globally unseen-map, independent replication, autonomous superiority or promotion claim. Reserved442 unplayed.')
    save(a.out,report)
    print(json.dumps(dict(roles=roles,criteria=criteria,pilot_gate_passed=report['pilot_gate_passed']),indent=2))


if __name__ == '__main__':
    main()
