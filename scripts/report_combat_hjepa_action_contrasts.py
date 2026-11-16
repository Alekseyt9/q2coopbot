"""Paired short-horizon effects of declared constant proposed-command interventions."""
import argparse
import collections
import json
import math
import pathlib
from train_combat_hjepa import sha, save
from prepare_combat_hjepa_effects import effect_vectors, WEIGHTS, effect_action
from audit_combat_hjepa_world_actions import primitive


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--matrix',type=pathlib.Path,required=True)
    ap.add_argument('--states',type=pathlib.Path,required=True)
    ap.add_argument('--spec',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    a=ap.parse_args();done=json.loads((a.matrix/'complete.json').read_text(encoding='utf-8-sig'))
    if done['state']!='complete' or not done['diagnostic_weapon_branches']:raise ValueError('completed diagnostic capture required')
    if sha(a.matrix/'plan.json')!=done['source_plan_sha256']:raise ValueError('changed frozen plan')
    plan=json.loads((a.matrix/'plan.json').read_text(encoding='utf-8-sig'))
    planned_seeds={x['engine_seed'] for task in plan['tasks'] for x in task['instances']}
    if not planned_seeds or any(task['split']!='train' for task in plan['tasks']):raise ValueError('TRAIN intervention plan required')
    meta=json.loads(pathlib.Path(str(a.states)+'.meta.json').read_text())
    if meta['complete_sha256']!=sha(a.matrix/'complete.json'):raise ValueError('changed completion')
    inputs={str(a.states):sha(a.states),str(a.spec):sha(a.spec),str(a.matrix/'complete.json'):sha(a.matrix/'complete.json'),str(a.matrix/'plan.json'):sha(a.matrix/'plan.json')}
    for field in ['sources_sha256','native_receipts_sha256']:
        for name,digest in meta[field].items():
            if sha(pathlib.Path(name))!=digest:raise ValueError('changed source proof')
            inputs[name]=digest
    spec=json.loads(a.spec.read_text());commands=spec['commands'];models={v['name']:v for v in done['models']}
    if set(models)!=set(commands):raise ValueError('wrong intervention set')
    for name,v in models.items():
        if set(v['seeds'])!=planned_seeds:raise ValueError('incomplete planned intervention coverage')
        if sha(pathlib.Path(v['model']))!=v['sha256']:raise ValueError('model changed')
        inputs[v['model']]=v['sha256']
        m=json.loads(pathlib.Path(v['model']).read_text());layer=m['motor']
        f,s,attack=commands[name]
        expected=[math.atanh(f),math.atanh(s),0.,0.,1. if attack else -1.,1.,0.,0.]
        if m['planner']!='bc' or m['weapon_mode']!='keep_equipped_v1' or len(layer)!=1 or layer[0]['bias']!=expected or any(any(row) for row in layer[0]['weight']):
            raise ValueError('not the declared constant intervention')
    grouped=collections.defaultdict(dict)
    for line in a.states.open():
        state=json.loads(line)
        if state['variant'] in grouped[state['seed']] or state['split']!='train':raise ValueError('duplicate/wrong state')
        grouped[state['seed']][state['variant']]=state
    if set(grouped)!=planned_seeds or sum(map(len,grouped.values()))!=len(planned_seeds)*len(commands):raise ValueError('incomplete intervention matrix')
    if meta['rows']!=sum(map(len,grouped.values())):raise ValueError('state export count changed')
    rows=[];rejected=[];omitted=[]
    for seed,arms in grouped.items():
        if set(arms)!=set(commands):raise ValueError('incomplete paired world')
        base=arms['idle'];baseline_fixture=None;baseline_entities=None
        if any(s['identity']!=base['identity'] or s['weapon_features']!=base['weapon_features'] for s in arms.values()):
            rejected.append(dict(seed=seed,reason='initial_causal_observation_mismatch'));continue
        for variant,state in arms.items():
            source=pathlib.Path(state['source']);report=json.loads(pathlib.Path(state['case_report']).read_text())
            if not report['capture_complete'] or not report['provenance_valid'] or len(report['results'])!=1:raise ValueError('invalid branch receipt')
            result=report['results'][0]
            if result['seed']!=seed or not result['capture_valid'] or not result['generated_start']['confirmed']:raise ValueError('branch binding changed')
            if baseline_fixture is None:
                baseline_fixture=result['generated_fixture'];baseline_entities=result['generated_start']['entities']
            elif result['generated_fixture']!=baseline_fixture or result['generated_start']['entities']!=baseline_entities:
                raise ValueError('native initial fixture/entities mismatch')
            steps=[json.loads(l) for l in source.open()]
            op=source.with_name('server_outcomes.jsonl');inputs[str(op)]=sha(op)
            vectors,ledger=effect_vectors(result,steps,[json.loads(l) for l in op.open()])
            frame=state['identity']['frame'];by_frame={s['observation']['identity']['frame']:s for s in steps}
            declared=[*commands[variant][:2],0.,0.,float(commands[variant][2]),1.,0.,0.]
            for horizon in [1,4,16]:
                chunk=[];terminal=False;control_transfer=False
                for current in range(frame,frame+horizon):
                    if current>=ledger['end_frame']:
                        terminal=ledger['terminal_step'] is not None
                        if not terminal:raise ValueError('right-censored diagnostic horizon')
                        break
                    s=by_frame[current]
                    if s['owner']!='provider':
                        omitted.append(dict(seed=seed,variant=variant,horizon=horizon,frame=current,reason='control_transfer',owner=s['owner']))
                        control_transfer=True;break
                    if not s.get('native_step') or not s['server_execution']['matched'] or s['native_step']['end_frame']!=s['native_step']['begin_frame']+1:
                        raise ValueError('intervention native mismatch')
                    if any(abs(v-w)>1e-12 for v,w in zip(primitive(s['action']),declared)) or s['action'].get('weapon','')!='':raise ValueError('captured command differs from intervention')
                    chunk.append(s)
                if control_transfer:continue
                effects=[sum(vectors[s['index']][k] for s in chunk) for k in range(6)]
                rows.append(dict(seed=seed,variant=variant,horizon=horizon,features=state['weapon_features'],
                     planned_actions=[effect_action(chunk[0]['action'])]*horizon,effects=effects,
                     reward=sum(v*w for v,w in zip(effects,WEIGHTS)),actual_ticks=len(chunk),absorbed_early_terminal=terminal,
                     guarded_ticks=sum(bool(s.get('interventions')) for s in chunk),
                     applied_changed_ticks=sum(any(abs(v-w)>1e-3 for v,w in zip(primitive(s['action']),primitive(s['applied_action']))) for s in chunk)))
    paired={}
    for horizon in [1,4,16]:
        samples={(r['seed'],r['variant']):r for r in rows if r['horizon']==horizon};comparisons={}
        for variant in commands:
            if variant=='idle':continue
            reference='idle' if variant=='fire' else 'fire'
            diffs=[samples[seed,variant]['reward']-samples[seed,reference]['reward'] for seed in grouped if (seed,variant) in samples and (seed,reference) in samples]
            comparisons[variant]=dict(reference=reference,paired_worlds=len(diffs),nonzero_reward_contrasts=sum(abs(x)>1e-9 for x in diffs),mean_reward_difference=sum(diffs)/len(diffs) if diffs else None)
        paired[horizon]=comparisons
    a.out.mkdir(exist_ok=False,parents=True)
    with (a.out/'rows.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    save(a.out/'report.json',dict(version='hjepa_controlled_command_contrasts_v1',promotion=False,
         matched_worlds=len(grouped)-len(rejected),rejected=rejected,omitted_horizons=omitted,rows=len(rows),rows_sha256=sha(a.out/'rows.jsonl'),paired=paired,sources_sha256=inputs,
         scope='Constant proposed-command policy interventions, with stock guards as environment. Same observed initial871features and verified native initial entities. No physical state restore or exclusivity after terminal claimed. Targets cover1/4/16 ticks, terminal absorbing. World-level paired contrasts, not independent branch worlds, learned-controller evaluation or optimal action labels.'))
    print(json.dumps(dict(matched_worlds=len(grouped)-len(rejected),rows=len(rows),paired=paired),indent=2))


if __name__=='__main__':main()
