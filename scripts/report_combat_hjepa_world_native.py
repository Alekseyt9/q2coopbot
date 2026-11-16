"""Native validation of proposed-input predictors against applied-input refresh and parent."""
import argparse
import collections
import json
import pathlib
from report_combat_hjepa_weapon_stochastic import load,summarize,bootstrap
from train_combat_hjepa import save


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--matrix',type=pathlib.Path,required=True)
    ap.add_argument('--forecast-gate',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    a=ap.parse_args();data,models,fixtures=load(a.matrix,worlds=16,split='validation')
    learned=['proposed','applied','parent'];controls=['keep','live-u1','rules']
    if set(data)!={f'{role}-s{s}' for role in learned for s in range(2)}|set(controls):
        raise ValueError('incomplete predictor ablation')
    for stream in range(2):
        source=models[f'parent-s{stream}']
        for role in learned:
            model=models[f'{role}-s{stream}'];ignored={'levels','world_action_space','weapon_sampling_seed'}
            if model['weapon_sampling_seed']!=261700+stream:raise ValueError('wrong coupled sampling seed')
            if {k:v for k,v in model.items() if k not in ignored}!={k:v for k,v in source.items() if k not in ignored}:
                raise ValueError('motor or decision head changed')
            for old,new in zip(source['levels'],model['levels']):
                if any(old[k]!=new[k] for k in ['encoder','utility','stride']):raise ValueError('frozen world component changed')
            if role=='proposed' and model.get('world_action_space')!='proposed_primitive_v1':raise ValueError('missing proposed contract')
    roles={role:summarize([r for s in range(2) for r in data[f'{role}-s{s}'].values()]) for role in learned}
    roles.update({role:summarize(list(data[role].values())) for role in controls})
    keys=['win','death','received_hp','self_hp','score'];comparisons={}
    for reference in ['applied','parent',*controls]:
        clusters=collections.defaultdict(list)
        for seed in fixtures:
            after={k:sum(data[f'proposed-s{s}'][seed][k] for s in range(2))/2 for k in keys}
            before={k:(sum(data[f'{reference}-s{s}'][seed][k] for s in range(2))/2 if reference in learned else data[reference][seed][k]) for k in keys}
            clusters[data['proposed-s0'][seed]['map']].append({k:after[k]-before[k] for k in keys})
        comparisons[reference]={k:bootstrap([sum(row[k] for row in group)/len(group) for group in clusters.values()]) for k in keys}
    p,a_ref,old=(roles[role]['totals'] for role in learned)
    combat=dict(win_gain_vs_applied=p['win']>=a_ref['win']+2,win_gain_vs_parent=p['win']>=old['win']+2,
                deaths_not_increased=p['death']<=min(a_ref['death'],old['death']),
                received_not_increased=p['received_hp']<=old['received_hp'],self_not_increased=p['self_hp']<=old['self_hp'])
    forecast=json.loads(a.forecast_gate.read_text())
    report=dict(version='hjepa_world_action_native_validation_v1',promotion=False,roles=roles,
        proposed_minus_reference=comparisons,combat_gate=dict(passed=all(combat.values()),criteria=combat),
        forecast_gate=forecast,combined_gate_passed=all(combat.values()) and forecast['passed'],
        scope='Only predictors refitted, encoders/utilities/actor/motor/weapon head unchanged. '
              '16newVALIDATION441worlds,2coupledstreams per model,16actual episodes per control. '
              'Bootstrap uses eight map clusters; no final test or autonomous superiority claim.')
    save(a.out,report);print(json.dumps(dict(roles=roles,combat_gate=combat,forecast_gate=forecast['criteria']),indent=2))


if __name__=='__main__':main()
