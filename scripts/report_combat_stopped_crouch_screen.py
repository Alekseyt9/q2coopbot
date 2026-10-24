"""Close a guard-only screen with paired outcomes, native pose and physics checks."""
import argparse
import collections
import copy
import json
import math
import os
from pathlib import Path
import subprocess

from evaluate_registered_combat_models import read, save, sha
from audit_combat_stopped_crouch_native_probe import normalized, same
from audit_combat_recorded_crouch_motion import SNAPSHOT_FIELDS
from report_combat_movement_variance_confirmation import paired_summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation', 'spec', 'quality', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists()
    a.out.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        h = pins.get(str(path)) or sha(path)
        assert expected is None or h == expected, str(path)
        pins[str(path)] = h
        return path

    pin(__file__)
    spec = read(pin(a.spec))
    closed = read(pin(a.evaluation/'complete.json'))
    assert closed['state']=='complete' and closed['native_cases']==24 and closed['cuda_cases']==16
    assert closed['spec_sha256']==sha(a.spec) and not closed['training_performed']
    fixture = read(pin(a.evaluation/'fixture-proof.json'))
    assert fixture['spec_sha256']==sha(a.spec)
    models = {m['label']: m for m in fixture['models']}
    assert set(models)=={'control','crouch'}
    assert models['control']['model_sha256']==models['crouch']['model_sha256']==spec['fixed_model_sha256']
    pool = read(pin(a.evaluation/'pool/report.json',closed['pool_sha256']))
    assert pool['source_unchanged'] and pool['usable_captures']==24
    labels = {i: Path(p['path']).stem.removesuffix('-plan') for i,p in enumerate(fixture['plans'])}
    qseal = read(pin(a.quality/'complete.json'))
    quality = read(pin(a.quality/'report.json',qseal['report_sha256']))
    totals, comparisons = {}, {}
    for item in closed['comparisons']:
        data = read(pin(a.evaluation/('quality-'+item['name']+'.json'),item['sha256']))
        assert item['after']=='crouch' and len(data['pairs'])==8
        totals['crouch'],totals[item['before']] = data['totals']['after'],data['totals']['before']
        comparisons[item['name']] = paired_summary(data['pairs'])
    assert set(totals)=={'control','crouch','rules'}
    ownership = read(pin(a.evaluation/'ownership-crouch.json'))
    assert ownership['pool_sha256']==closed['pool_sha256'] and len(ownership['cases'])==8
    entries, samples, traces, intervention_rows = [], {}, {}, []
    pose_counts = collections.defaultdict(collections.Counter)
    for job in pool['jobs']:
        label = labels[job['plan_index']]
        if label=='rules':
            continue
        capture = read(pin(Path(job['root'])/'report.json'))
        result = capture['results'][0]
        assert capture['capture_complete'] and capture['provenance_valid'] and result['dispatch_valid']
        worker = Path(result['root'])
        manifest = read(pin(Path(job['root'])/'manifest.json'))
        assert bool(manifest.get('diagnostic_stopped_crouch_all_frames'))==(label=='crouch')
        folder = a.evaluation/('processing-'+label)/f"case-{job['job']:04d}"
        cm = read(pin(folder/'cuda/report.json'))
        proof = read(pin(folder/'cuda/cuda-verification.json',cm['cuda_verification_sha256']))
        assert proof['state']=='passed' and proof['device']=='cuda'
        assert bool(cm.get('diagnostic_only'))==(label=='crouch')
        if label=='crouch':
            assert cm['diagnostic_stopped_crouch_all_frames'] and not cm['ppo_training_ready']
        rows = {}
        for raw in map(json.loads,pin(worker/'bot.jsonl').open(encoding='utf-8-sig')):
            c = raw.get('combat_policy') or {}
            o = c.get('observation') or {}
            identity = o.get('identity') or {}
            if identity.get('life')!=1 or o.get('health',0)<=0 or (c.get('selection') or {}).get('owner')!='provider':
                continue
            frame = identity['frame']
            assert frame not in rows
            rows[frame] = raw
        assert rows
        traces[label,job['seed']] = rows
        for frame,raw in rows.items():
            c = raw['combat_policy']; sent = raw['sent_command']
            counts = pose_counts[label]
            counts['frames']+=1
            counts['requested_crouch']+=c['proposed_command']['Up']<0
            counts['applied_crouch']+=sent['Up']<0
            counts['standing_no_horizontal_command']+=sent['Forward']==sent['Side']==sent['Up']==0
            tags = [i for i in c['selection'].get('interventions') or [] if i['reason']=='diagnostic_checked_stopped_crouch']
            if not tags:
                continue
            assert label=='crouch' and len(tags)==1 and raw['on_ground'] and sent['Forward']==sent['Side']==0
            assert sent['Up']==c['proposed_command']['Up']<0 and c['applied_command']==sent
            counts['restored_crouch']+=1
            cleanup = read(pin(worker/'runtime/pak-cleanup.json'))
            assets = [pin(x['source'],x['sha256'].lower()) for x in cleanup['removed']]
            assert len(assets)==3 and len({p.parent for p in assets})==1
            asset_hashes = {p.name: pins[str(p)] for p in assets}
            assert all(asset_hashes[x['path']]==x['sha256'].lower() for x in manifest['bsp_assets'])
            before = copy.deepcopy(sent); before['Up']=0
            key = str(job['seed'])+'/'+str(frame)
            entries.append(dict(id=key,runtime_root=str(assets[0].parent),
                snapshot={k:raw[k] for k in SNAPSHOT_FIELDS if k in raw},
                proposed=c['proposed_command'],applied=before))
            samples[key] = dict(seed=job['seed'],frame=frame,sent=sent,row=raw,next=rows.get(frame+1))
    prefix_checks = []
    for (label,seed),rows in traces.items():
        if label!='crouch': continue
        baseline = traces['control',seed]
        changed = [f for f,r in rows.items() if any(i['reason']=='diagnostic_checked_stopped_crouch'
                    for i in r['combat_policy']['selection'].get('interventions') or [])]
        first = min(changed) if changed else None
        frames = sorted(f for f in rows if first is None or f<=first)
        assert frames
        ok = all(f in baseline for f in frames)
        for f in frames:
            if f not in baseline: continue
            def sig(r):
                c=r['combat_policy']
                return normalized(dict(observation=c['observation'],proposed=c['proposed'],
                    proposed_command=c['proposed_command'],sample=c['selection'].get('sample')))
            ok = ok and same(sig(rows[f]),sig(baseline[f]),1e-9)
            expected = copy.deepcopy(baseline[f]['sent_command'])
            if f==first: expected['Up']=rows[f]['sent_command']['Up']
            ok = ok and rows[f]['sent_command']==expected
        if first is None: ok = ok and set(rows)==set(baseline)
        prefix_checks.append(dict(seed=seed,first_intervention=first,frames=len(frames),passed=ok))
    physics_checks, position_errors, velocity_errors = [], [], []
    if entries:
        save(a.out/'geometry-request.json',dict(version='combat_recorded_motion_request_v1',entries=entries))
        binary = a.out.resolve()/'q2motion-audit.exe'
        env = dict(os.environ,GOCACHE=str(repo/'workspace/build/gocache'),GOTOOLCHAIN='auto')
        for label,command in [('build',['go','build','-o',binary,'./cmd/q2motion-audit']),
            ('geometry',[binary,'--request',(a.out/'geometry-request.json').resolve(),'--out',(a.out/'geometry.json').resolve()])]:
            with (a.out/(label+'.log')).open('w',encoding='utf-8') as log:
                result=subprocess.run(list(map(str,command)),cwd=repo,env=env,stdout=log,stderr=log)
            assert result.returncode==0,str(a.out/(label+'.log'))
        for result in read(pin(a.out/'geometry.json'))['results']:
            s=samples[result['id']]; audit=result['audit']; pred=audit['stopped_crouch_prediction']; nxt=s['next']; raw=s['row']
            checked=bool(audit['stopped_crouch_eligible'] and audit['stopped_crouch_path_checked'] and pred
                         and audit['stopped_crouch_command']==s['sent'])
            native_comparable=bool(nxt and nxt['on_ground'] and nxt['health']==raw['health'] and nxt['armor']==raw['armor'])
            pe=ve=None
            if native_comparable and pred:
                pe=math.dist(nxt['self'],pred['position']); ve=math.dist(nxt['self_velocity'],pred['velocity'])
                position_errors.append(pe);velocity_errors.append(ve)
            native_ducked=bool(nxt and nxt['ducked'])
            physics_checks.append(checked and (pe is None or pe<=spec['mechanical_gate']['position_tolerance'])
                                  and (ve is None or ve<=spec['mechanical_gate']['velocity_tolerance']) and (not native_comparable or native_ducked))
            intervention_rows.append(dict(seed=s['seed'],frame=s['frame'],path_checked=checked,native_comparable=native_comparable,
                native_ducked=native_ducked,position_error=pe,velocity_error=ve))
    gate=spec['expansion_gate']; decision=totals['crouch']
    checks=dict(exposed=len(entries)>0,prefix_all_pairs=len(prefix_checks)==8 and all(x['passed'] for x in prefix_checks),
        restored_physics_all=bool(physics_checks) and all(physics_checks),comparable_native_frames=bool(position_errors),
        goals_control_plus_two=decision['goal']>=totals['control']['goal']+gate['additional_goals_over_control'],
        goals_at_least_rules=decision['goal']>=totals['rules']['goal'],
        observed_combat_owned_by_model=all(ownership['totals'][k]==0 for k in
            ('visible_enemy_rules_frames','visible_enemy_rules_attack_frames','visible_rules_missing_applied_frames','unknown_observation_rows')))
    for label in ('control','rules'):
        checks['deaths_no_more_than_'+label]=decision['death']<=totals[label]['death']
        checks['received_within_110_percent_'+label]=10*decision['received']<=11*totals[label]['received']
    assert all(sha(p)==h for p,h in pins.items())
    save(a.out/'report.json',dict(state='complete',version='combat_stopped_crouch_screen_result_v1',
        gate_passed=all(checks.values()),checks=checks,totals=totals,comparisons=comparisons,
        pose_counts={k:dict(v) for k,v in pose_counts.items()},prefix_checks=prefix_checks,interventions=intervention_rows,
        native_comparable=len(position_errors),position_max=max(position_errors,default=None),velocity_max=max(velocity_errors,default=None),
        actions=quality['variants'],inputs=pins,neural_execution=False,training_performed=False,promotion=False,
        scope='Same fixed weights, fresh paired 8-recipe screen. Guard-only intervention, no fitting, no general-superiority or campaign/coop claim.'))
    lines=['# Stopped crouch screen','','| Variant | Goals / 8 | Deaths | Kills | Received HP |','|---|---:|---:|---:|---:|']
    for label in ('control','crouch','rules'):
        t=totals[label];lines.append(f"| {label} | {t['goal']} | {t['death']} | {t['kills']} | {t['received']} |")
    lines+=['','Expansion gate: '+str(all(checks.values()))+'. Promotion: false.']
    (a.out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json'),markdown_sha256=sha(a.out/'results.md')))
    print(dict(gate_passed=all(checks.values()),totals=totals,checks=checks,restored=len(entries)))


if __name__=='__main__':
    main()
