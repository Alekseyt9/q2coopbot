"""Paired categorical validation of two credit estimators and their frozen source."""
import argparse
import collections
from report_combat_hjepa_weapon_stochastic import load,summarize,bootstrap
from train_combat_hjepa import save


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--matrix',required=True,type=__import__('pathlib').Path)
    ap.add_argument('--out',required=True,type=__import__('pathlib').Path)
    a=ap.parse_args();data,models,fixtures=load(a.matrix,worlds=16,split='validation')
    learned=['rtg','episode','parent'];controls=['keep','live-u1','rules']
    expected={f'{role}-s{s}' for role in learned for s in range(2)}|set(controls)
    if set(data)!=expected:raise ValueError('missing estimator/control variants')
    for stream in range(2):
        parent=models[f'parent-s{stream}'];ignored={'weapon_actor','weapon_sampling_seed'}
        for role in learned:
            model=models[f'{role}-s{stream}']
            if model['weapon_sampling_seed']!=261600+stream:raise ValueError('wrong paired stream')
            if {k:v for k,v in model.items() if k not in ignored}!={k:v for k,v in parent.items() if k not in ignored}:
                raise ValueError('non-weapon fields differ')
    roles={role:summarize([r for s in range(2) for r in data[f'{role}-s{s}'].values()]) for role in learned}
    roles.update({role:summarize(list(data[role].values())) for role in controls})
    comparisons={};keys=['win','death','received_hp','self_hp','score']
    for reference in ['episode','parent',*controls]:
        maps=collections.defaultdict(list)
        for seed in fixtures:
            after={k:sum(data[f'rtg-s{s}'][seed][k] for s in range(2))/2 for k in keys}
            before={k:(sum(data[f'{reference}-s{s}'][seed][k] for s in range(2))/2
                       if reference in learned else data[reference][seed][k]) for k in keys}
            maps[data['rtg-s0'][seed]['map']].append({k:after[k]-before[k] for k in keys})
        comparisons[reference]={k:bootstrap([sum(v[k] for v in values)/len(values) for values in maps.values()]) for k in keys}
    r,e,p=(roles[role]['totals'] for role in learned)
    gate=dict(win_gain_at_least_two=r['win']>=e['win']+2,not_fewer_than_parent=r['win']>=p['win'],
              deaths_not_increased=r['death']<=min(e['death'],p['death']),
              received_not_increased=r['received_hp']<=p['received_hp'],self_not_increased=r['self_hp']<=p['self_hp'])
    save(a.out,dict(version='hjepa_credit_estimator_validation_v1',promotion=False,roles=roles,
                   rtg_minus_reference=comparisons,screen_gate=dict(passed=all(gate.values()),criteria=gate),
                   scope='Frozen categorical comparison on16newVALIDATIONworlds,2coupledstreams each '
                         '(32native episodes per learned role),16native episodes per control. '
                         'Intervals resample eight map clusters, not32independent worlds. '
                         'No held-out geometry or final superiority claim.'))
    print(__import__('json').dumps(dict(roles=roles,gate=gate),indent=2))


if __name__=='__main__':main()
