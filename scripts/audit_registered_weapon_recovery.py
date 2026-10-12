"""Observe recovery choices against the visible inventory and network release clock.

This is metadata analysis of sealed traces, not inference, reward relabeling,
or proof of shots. Model availability uses the existing observation-only oracle.
"""
import argparse
import collections
import pathlib
import json

from audit_registered_weapon_choices import audit_frames
from combat_equipped_weapon_features import equipped_weapon_index, CATEGORIES
from combat_weapon_features import inventory_tail
from process_combat_architecture_pool import read, save, sha


def summarize(frames):
    counts=collections.Counter(); requests=collections.Counter(); equipped=collections.Counter()
    first=None; first_request=None; first_supported=None; initial=None
    for obs,applied,owner in frames:
        if owner!='provider': continue
        assert applied is not None
        frame=obs['identity']['frame']; current=equipped_weapon_index(obs)
        mask=inventory_tail(obs)[19:31]
        available=[i for i in range(1,5) if mask[i]]
        clear=any(e.get('clear_shot') for e in obs.get('enemies') or [])
        request=applied.get('weapon') or ''
        requested=equipped_weapon_index(dict(weapon=request)) if request else 0
        if request: assert requested>0
        if first is None:
            first=frame
            initial=dict(frame=frame,equipped=CATEGORIES[current],inventory_known=bool(obs.get('inventory') is not None),
                         inventory_age_frames=obs.get('inventory_age_frames'),supported_available=[CATEGORIES[i] for i in available])
        if request and first_request is None: first_request=frame
        if current in range(1,5) and first_supported is None: first_supported=frame
        counts['provider_frames']+=1; equipped[CATEGORIES[current]]+=1
        requests[CATEGORIES[requested] if requested else 'keep']+=1
        if available:
            counts['supported_available_frames']+=1
            if current not in range(1,5):
                counts['outside_with_supported_available_frames']+=1
                counts['outside_available_keep_frames']+=int(not request)
                counts['outside_available_supported_request_frames']+=int(requested in range(1,5))
                if clear:
                    counts['clear_outside_available_frames']+=1
                    counts['clear_outside_available_keep_frames']+=int(not request)
        counts['inventory_unavailable_frames']+=int(not available)
    assert first is not None
    return dict(counts=dict(counts),equipped=dict(equipped),requests=dict(requests),initial=initial,
                first_request_delay_frames=None if first_request is None else first_request-first,
                first_supported_delay_frames=None if first_supported is None else first_supported-first,
                observed_supported=first_supported is not None,choice_audit=audit_frames(frames))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evaluation',type=pathlib.Path,required=True)
    ap.add_argument('--spec',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    ap.add_argument('--labels',help='Comma-separated learned variant labels; default preserves historical after-only selection')
    args=ap.parse_args(); assert not args.out.exists()
    root=args.evaluation.resolve(); seal=read(root/'complete.json'); spec=read(args.spec)
    assert seal['state']=='complete' and not seal['training_performed']
    assert sha(args.spec)==seal['spec_sha256']
    selected=set(args.labels.split(',')) if args.labels else None
    if selected is not None:
        assert len(selected)==len(args.labels.split(',')) and '' not in selected
        assert selected <= {v['label'] for v in spec['variants'] if v['mode']=='learned'}
    pool=read(root/'pool/report.json'); assert sha(root/'pool/report.json')==seal['pool_sha256']
    assert pool['state']=='complete' and pool['source_unchanged']
    pins={root/'complete.json':sha(root/'complete.json'),args.spec:sha(args.spec),root/'pool/report.json':seal['pool_sha256']}
    variants={}
    for index,variant in enumerate(spec['variants']):
        label=variant['label']
        if variant['mode']!='learned': continue
        if selected is not None:
            if label not in selected: continue
        elif '-after-' not in label: continue
        plan_path=root/(label+'-plan.json'); plan=read(plan_path); pins[plan_path]=sha(plan_path)
        assert pins[plan_path]==pool['plans'][index]['sha256']
        execution_path=root/('execution-'+label+'.json'); execution=read(execution_path); pins[execution_path]=sha(execution_path)
        assert execution['state']=='complete' and execution['pool_sha256']==seal['pool_sha256']
        assert execution['selected_plan_index']==index
        bindings={c['job']:c for c in execution['cases']}; cases=[]
        for job in pool['jobs']:
            if job['plan_index']!=index: continue
            episode=plan['tasks'][job['task_index']]['episode']['id']
            if not episode.endswith('-recovery'): continue
            binding=bindings[job['job']]; capture_path=pathlib.Path(job['root'])/'report.json'
            assert not job['error'] and sha(capture_path)==binding['capture_sha256']
            capture=read(capture_path); result=capture['results'][0]
            trace=pathlib.Path(result['root'])/'bot.jsonl'; assert sha(trace)==binding['trace_sha256']
            frames=[]; seen=set()
            with trace.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    control=json.loads(line).get('combat_policy') or {}
                    owner=(control.get('selection') or {}).get('owner')
                    if not owner: continue
                    obs=control['observation']; identity=obs['identity']
                    if identity['life']!=1 or obs['health']<=0 or identity['frame']<binding['network_release_frame']: continue
                    assert identity['spawncount']==binding['release_generation']
                    key=tuple(identity[k] for k in ('map','spawncount','connection','actor','life','frame'))
                    assert key not in seen; seen.add(key)
                    frames.append((obs,control.get('applied'),owner))
            value=summarize(frames)
            assert sha(trace)==binding['trace_sha256']
            value.update(job=job['job'],episode=episode,seed=job['seed'],goal=bool(result['goal_stop']),
                         trace_sha256=binding['trace_sha256'],capture_sha256=binding['capture_sha256'])
            cases.append(value)
        expected=sum(len(t['seeds']) for t in plan['tasks'] if t['episode']['id'].endswith('-recovery'))
        expected_jobs={j['job'] for j in pool['jobs'] if j['plan_index']==index and
            plan['tasks'][j['task_index']]['episode']['id'].endswith('-recovery')}
        assert expected>0 and len(cases)==len(expected_jobs)==expected
        assert {c['job'] for c in cases}==expected_jobs
        counts=dict(sum((collections.Counter(c['counts']) for c in cases),collections.Counter()))
        variants[label]=dict(cases=cases,counts=counts,wins=sum(c['goal'] for c in cases),
            cases_with_observed_supported=sum(c['observed_supported'] for c in cases),
            cases_without_request=sum(c['first_request_delay_frames'] is None for c in cases),
            requests=dict(sum((collections.Counter(c['requests']) for c in cases),collections.Counter())))
        print(label,{k:v for k,v in variants[label].items() if k not in ('cases','counts','requests')},flush=True)
    for p,digest in pins.items(): assert sha(p)==digest,str(p)
    save(args.out,dict(state='complete',version='registered_weapon_recovery_observation_v1',
         pool_sha256=seal['pool_sha256'],evaluation_sha256=pins[root/'complete.json'],script_sha256=sha(__file__),
         availability_oracle_sha256=sha(pathlib.Path(__file__).with_name('combat_weapon_features.py')),
         variants=variants,training_performed=False,promotion=False,
         selected_labels=sorted(variants),
         scope='Selected learned variants, held-out recovery episodes only. Visible inventory age/mask and observed equipment; correct network release clock. Requests are commands, not completed dispatch or physical shots. No inference, fitting or causal attribution.'))


if __name__=='__main__': main()
