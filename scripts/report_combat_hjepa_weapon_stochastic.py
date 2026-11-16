"""Compare the trained categorical policy with its source on paired TRAIN worlds."""
import argparse
import json
import pathlib
import random
from train_combat_hjepa import sha, save
from train_combat_hjepa_weapon_onpolicy import episode_return


def load(root,worlds=8,split='train'):
    complete=json.loads((root/'complete.json').read_text(encoding='utf-8-sig'))
    if complete['state']!='complete' or complete['source_plan_sha256']!=sha(root/'plan.json'):
        raise ValueError('complete immutable matrix required')
    plan=json.loads((root/'plan.json').read_text(encoding='utf-8-sig'))
    fixtures={x['engine_seed']:x for t in plan['tasks'] for x in t['instances']}
    if len(fixtures)!=worlds or any(x['split']!=split for x in fixtures.values()):
        raise ValueError('wrong diagnostic world count/split')
    results={};models={}
    for v in complete['models']:
        if v['mode']=='learned':
            path=pathlib.Path(v['model'])
            if sha(path)!=v['sha256']:raise ValueError('model changed')
            models[v['name']]=json.loads(path.read_text())
        rows={}
        for p in (root/v['name']).glob('case-*/report.json'):
            report=json.loads(p.read_text())
            if not report['capture_complete'] or not report['provenance_valid']:raise ValueError('invalid native case')
            for r in report['results']:
                seed=r['seed']
                if seed in rows or not r['capture_valid'] or not r['dataset']['command_proof']['accepted']:
                    raise ValueError('duplicate/unverified native episode')
                if r['generated_fixture']!=fixtures[seed] or not r['generated_start']['confirmed']:
                    raise ValueError('fixture/start mismatch')
                damage=r['first_life']['damage']
                death=bool(r['death_stop']);win=bool(r['goal_stop']) and not death
                rows[seed]=dict(map=r['first_life']['map'],win=int(win),death=int(death),timeout=int(not win and not death),
                    received_hp=damage['received_health_damage'],self_hp=damage['self_health_damage'],
                    score=episode_return(r))
        if set(rows)!=set(fixtures):raise ValueError('missing world')
        results[v['name']]=rows
    return results,models,fixtures


def summarize(rows):
    keys=['win','death','timeout','received_hp','self_hp','score']
    return dict(native_episodes=len(rows),totals={k:sum(r[k] for r in rows) for k in keys},
                per_episode={k:sum(r[k] for r in rows)/len(rows) for k in keys})


def bootstrap(values):
    rng=random.Random(261502)
    draws=sorted(sum(rng.choice(values) for _ in values)/len(values) for _ in range(10000))
    return dict(mean=sum(values)/len(values),approx_95_interval=[draws[250],draws[9749]])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--initial-matrix',type=pathlib.Path,required=True)
    ap.add_argument('--final-probe',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    a=ap.parse_args()
    initial,imodels,fixtures=load(a.initial_matrix)
    final,fmodels,other=load(a.final_probe)
    if fixtures!=other:raise ValueError('different paired cohorts')
    for stream in range(4):
        left=imodels[f'initial-s{stream}'];right=fmodels[f'warm-0-s{stream}']
        if left['weapon_sampling_seed']!=261400+stream or right['weapon_sampling_seed']!=261400+stream:
            raise ValueError('wrong coupled random stream')
        ignored={'weapon_actor','weapon_sampling_seed'}
        if {k:v for k,v in left.items() if k not in ignored}!={k:v for k,v in right.items() if k not in ignored}:
            raise ValueError('non-weapon behavior differs')
    roles=dict(initial_categorical=[r for s in range(4) for r in initial[f'initial-s{s}'].values()],
               final_categorical=[r for s in range(4) for r in final[f'warm-0-s{s}'].values()])
    controls=['initial-greedy','final-greedy','keep','live-u1','rules']
    roles.update({name:list(initial[name].values()) for name in controls})
    keys=['win','death','received_hp','self_hp','score']
    paired=[]
    for seed in sorted(fixtures):
        before=[initial[f'initial-s{s}'][seed] for s in range(4)]
        after=[final[f'warm-0-s{s}'][seed] for s in range(4)]
        delta={k:sum(r[k] for r in after)/4-sum(r[k] for r in before)/4 for k in keys}
        paired.append(dict(seed=seed,final_minus_initial=delta))
    comparisons={}
    for name in controls:
        comparisons[name]={k:bootstrap([sum(final[f'warm-0-s{s}'][seed][k] for s in range(4))/4-
                                       initial[name][seed][k] for seed in sorted(fixtures)]) for k in keys}
    save(a.out,dict(version='hjepa_weapon_stochastic_diagnostic_v1',promotion=False,
        roles={name:summarize(rows) for name,rows in roles.items()},paired_world_differences=paired,
        final_minus_initial={k:bootstrap([v['final_minus_initial'][k] for v in paired]) for k in keys},
        final_categorical_minus_controls=comparisons,
        scope='TRAIN-only reused eight-world diagnostic. Categorical policies each have four coupled streams '
              '(32 actual episodes); each deterministic/controller reference has eight actual episodes, not '
              '32 invented reruns. Intervals resample eight worlds, never treating 32 streams as 32 independent '
              'worlds. Tiny diagnostic, no final quality or promotion claim.'))
    print(json.dumps({name:summarize(rows) for name,rows in roles.items()},indent=2))


if __name__=='__main__':main()
