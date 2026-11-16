"""Matched native screen of full primitive CG versus Jacobi-PCG fitting."""
import argparse
import collections
import copy
import json
import pathlib
from train_combat_hjepa import save,sha
from report_combat_hjepa_weapon_stochastic import load,summarize,bootstrap
from report_combat_hjepa_categorical import native_cycles


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=pathlib.Path,required=True)
    a = ap.parse_args();a.root = a.root.resolve()
    data,models,fixtures = load(a.root/'native-validation',worlds=16,split='validation')
    control,_,other_fixtures = load(a.root/'rules-validation',worlds=16,split='validation')
    expected = {f'{role}-s{s}' for role in ['parent','cg','pcg'] for s in range(2)}|{'source-mean','cg-mean','pcg-mean','live-u1'}
    if set(data) != expected or set(control) != {'rules'} or fixtures != other_fixtures:
        raise ValueError('incomplete matched solver cohort')
    data.update(control)
    for stream in range(2):
        parent = models[f'parent-s{stream}']
        for role in ['parent','cg','pcg']:
            model = copy.deepcopy(models[f'{role}-s{stream}'])
            if model['primitive_sampling']['seed'] != 262600+stream or model['primitive_sampling']['categorical_temperature'] != 1.:
                raise ValueError('changed coupled distribution')
            model['levels'][0]['encoder'] = parent['levels'][0]['encoder'];model['actor'] = parent['actor']
            model['primitive_sampling']['log_std'] = parent['primitive_sampling']['log_std']
            if model != parent:raise ValueError('nonprimitive model changed')
    for mean,role in [('source-mean','parent'),('cg-mean','cg'),('pcg-mean','pcg')]:
        model = copy.deepcopy(models[role+'-s0']);model.pop('primitive_sampling')
        if model != models[mean]:raise ValueError('mean weights differ')
    cg_fit = json.loads((a.root/'cg/report.json').read_text());pcg_fit = json.loads((a.root/'pcg/report.json').read_text())
    if cg_fit['binding'] != pcg_fit['binding'] or cg_fit['admission'] != pcg_fit['admission']:
        raise ValueError('different training samples or credit')
    def common(algorithm):return {k:v for k,v in algorithm.items() if k not in ['optimizer','preconditioner']}
    if common(cg_fit['algorithm']) != common(pcg_fit['algorithm']) or any(len(r['history']) != 4 or r['stopped'] for r in [cg_fit,pcg_fit]):
        raise ValueError('different budgets, objective or failed nonlinear fit')
    roles = {role:summarize([r for s in range(2) for r in data[f'{role}-s{s}'].values()]) for role in ['parent','cg','pcg']}
    roles.update({role:summarize(list(data[role].values())) for role in ['source-mean','cg-mean','pcg-mean','live-u1','rules']})
    criteria = {}
    for mode,p,cg,pcg,gain in [('sampled',roles['parent']['totals'],roles['cg']['totals'],roles['pcg']['totals'],4),
                              ('primitive_mean',roles['source-mean']['totals'],roles['cg-mean']['totals'],roles['pcg-mean']['totals'],2)]:
        criteria[mode] = dict(win_gain_vs_parent=pcg['win'] >= p['win']+gain,wins_not_below_cg=pcg['win'] >= cg['win'],
            deaths_not_increased=pcg['death'] <= min(p['death'],cg['death']),
            received_not_increased=pcg['received_hp'] <= min(p['received_hp'],cg['received_hp']),
            self_not_increased=pcg['self_hp'] <= min(p['self_hp'],cg['self_hp']))
    comparisons = {}
    keys = ['win','death','received_hp','self_hp','score']
    for reference in ['parent','cg']:
        clusters = collections.defaultdict(list)
        for seed in fixtures:
            clusters[data['pcg-s0'][seed]['map']].append({k:sum(data[f'pcg-s{s}'][seed][k]-data[f'{reference}-s{s}'][seed][k] for s in range(2))/2 for k in keys})
        comparisons['pcg-minus-'+reference] = {k:bootstrap([sum(v[k] for v in group)/len(group) for group in clusters.values()]) for k in keys}
    cycles,hashes = {},{}
    for role in expected-{'live-u1'}:
        cycles[role],bound = native_cycles(a.root/'native-validation',role);hashes.update(bound)
    for role in ['parent','cg','pcg']:
        cycles[role] = {k:sum(cycles[f'{role}-s{s}'][k] for s in range(2)) for k in cycles[f'{role}-s0']}
    protocol = json.loads((a.root/'protocol.json').read_text())
    report = dict(version='hjepa_jacobi_pcg_native_screen_v1',promotion=False,roles=roles,criteria=criteria,
        pilot_gate_passed=all(all(v.values()) for v in criteria.values()),cycles=cycles,cycle_source_sha256=hashes,
        paired_map_cluster_intervals=comparisons,protocol_sha256=sha(a.root/'protocol.json'),
        scope=f'176actualgames,16fresh validation{protocol["validation"]["offset"]}worlds,eight knownmaps. Sampled roles32games/two coupled streams,mean/controller references16each. CG andPCG fit identical knownTRAIN508round1 samples,4updates each. Analytic diagonal construction adds cost beyond40HVPiteration budget. MeanKL and nonlinear surrogate checks unchanged; no convergence guarantee. Weapon/world frozen. Ruleslegacy_v1 dispatch differs from learnedstock_weapons_fire_v1. No unseen-map, independent replication or promotion claim; reserved442unplayed.')
    save(a.root/'native-screen.json',report)
    print(json.dumps(dict(roles=roles,criteria=criteria,cycles={k:cycles[k] for k in ['parent','cg','pcg','source-mean','cg-mean','pcg-mean']},pilot_gate_passed=report['pilot_gate_passed']),indent=2))


if __name__=='__main__':main()
