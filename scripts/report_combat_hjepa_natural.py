"""Matched fresh-world Adam versus natural primitive optimizer screen."""
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
    ap.add_argument('--rules-matrix', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    a = ap.parse_args()
    data, models, fixtures = load(a.matrix, worlds=16, split='validation')
    controls, _, control_fixtures = load(a.rules_matrix, worlds=16, split='validation')
    expected = {'parent','adam','natural','parent-sampled','adam-sampled','natural-sampled','live-u1'}
    if set(data) != expected or set(controls) != {'rules'} or fixtures != control_fixtures:
        raise ValueError('incomplete or different validation cohort')
    data.update(controls)
    parent = models['parent-sampled']
    for role in ('parent','adam','natural'):
        model = copy.deepcopy(models[role+'-sampled'])
        if model['primitive_sampling']['seed'] != 261990:
            raise ValueError('uncoupled primitive stream')
        model.pop('primitive_sampling')
        if model != models[role]:
            raise ValueError('sampled/mean weights differ')
        model = copy.deepcopy(models[role+'-sampled'])
        model['levels'][0]['encoder'] = parent['levels'][0]['encoder']
        model['actor'] = parent['actor']
        model['primitive_sampling']['log_std'] = parent['primitive_sampling']['log_std']
        if model != parent:
            raise ValueError('nonprimitive model changed')
    roles = {name:summarize(list(values.values())) for name, values in data.items()}
    keys = ['win','death','received_hp','self_hp','score']
    comparisons = {}
    for suffix in ('','-sampled'):
        for reference in ('parent','adam'):
            candidate, ref = 'natural'+suffix, reference+suffix
            clusters = collections.defaultdict(list)
            for seed in fixtures:
                clusters[data[candidate][seed]['map']].append({k:data[candidate][seed][k]-data[ref][seed][k] for k in keys})
            comparisons[candidate+'-minus-'+ref] = {k:bootstrap([sum(r[k] for r in group)/len(group) for group in clusters.values()]) for k in keys}
    criteria = {}
    for suffix in ('','-sampled'):
        parent_totals, adam, natural = (roles[role+suffix]['totals'] for role in ('parent','adam','natural'))
        criteria['sampled' if suffix else 'primitive_mean'] = dict(
            win_gain_vs_parent=natural['win'] >= parent_totals['win']+2,
            wins_not_below_adam=natural['win'] >= adam['win'],
            deaths_not_increased=natural['death'] <= min(parent_totals['death'],adam['death']),
            received_not_increased=natural['received_hp'] <= min(parent_totals['received_hp'],adam['received_hp']),
            self_not_increased=natural['self_hp'] <= min(parent_totals['self_hp'],adam['self_hp']))
    report = dict(version='hjepa_natural_primitive_screen_v1',promotion=False,roles=roles,
        paired_map_cluster_intervals=comparisons,criteria=criteria,
        pilot_gate_passed=all(all(v.values()) for v in criteria.values()),
        scope='128actualgames,16fresh validation445worlds,eight knownmaps. Adam/natural fitted on same fresh TRAIN509batch with four full-batch updates each. One coupled primitive stream and primitive-mean evaluation, same frozen weapon seed. Rules uses supported legacy_v1 dispatch; learned variants stock_weapons_fire_v1. No held-out-map, independent replication, or promotion claim; reserved442unplayed.')
    save(a.out, report)
    print(json.dumps(dict(roles=roles,criteria=criteria,pilot_gate_passed=report['pilot_gate_passed']),indent=2))


if __name__ == '__main__':
    main()
