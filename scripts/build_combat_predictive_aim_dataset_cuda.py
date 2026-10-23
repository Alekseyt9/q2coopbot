"""Native randomized-aim labels + causal features; predictive-only, not PPO."""
import argparse
import collections
import gzip
import json
import math
import subprocess
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from native_combat_release import release_boundary
from report_combat_machinegun_hits import scan
from report_registered_machinegun_outcomes import outcome

MASK=(1<<64)-1
OFFSETS=[0,-5,-2.5,2.5,5]


def mix(x):
    x=((x^(x>>30))*0xbf58476d1ce4e5b9)&MASK
    x=((x^(x>>27))*0x94d049bb133111eb)&MASK
    return x^(x>>31)


def candidate_for(seed,frame,life):
    word=mix((seed^(frame*0x9e3779b97f4a7c15)^(life*0xd1b54a32d192ed03))&MASK)
    return 0 if word>>63==0 else 1+mix(word^0x8cb92baa3f3d8dd7)%24


def audit_sample(step,seed):
    sample=step.get('sample') or {};r=sample.get('aim_exploration')
    if r is None:return None
    o=step['observation'];assert sample['sampling_seed']==seed
    index=candidate_for(seed,o['identity']['frame'],o['identity']['life'])
    assert r['candidates']==25 and r['candidate']==index
    scale=15 if sample.get('aim_mode',0)==1 else 180
    old=[math.tanh(v)*scale for v in r['source_latent']]
    expected=old[:]
    if index:
        for axis,offset in enumerate((OFFSETS[index//5],OFFSETS[index%5])):
            expected[axis]=max(-1+1e-6,min(1-1e-6,(old[axis]+offset)/scale))*scale
    actual=[math.tanh(v)*scale for v in sample['latent'][2:4]]
    assert max(abs(a-b) for a,b in zip(expected,actual))<1e-5
    action=step['action']
    assert max(abs(action[k]-actual[i]) for i,k in enumerate(('yaw_delta_degrees','pitch_delta_degrees')))<1e-4
    return dict(candidate=index,intervened=index!=0,effective=max(abs(a-b) for a,b in zip(actual,old))>.001)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','exporter','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();assert not args.out.exists();root=args.capture.resolve();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path);assert expected is None or expected==digest,str(path)
        pins[str(path)]=digest;return path
    closed=read(pin(root/'native-complete.json'))
    recipe=read(pin(root/'recipe.json',closed['recipe_sha256']))
    assert recipe['purpose']=='shot_predictive_train_only' and recipe['split']=='train' and not recipe['ppo_ready']
    pool=read(pin(root/'pool/report.json',closed['pool_sha256']))
    assert pool['state']=='complete' and pool['source_unchanged'] and not any(j['error'] for j in pool['jobs'])
    execution=read(pin(root/'execution.json',closed['execution_sha256']))
    assert execution['state']=='complete' and execution['pool_sha256']==closed['pool_sha256']
    bindings={c['job']:c for c in execution['cases']}
    plans=[read(pin(p['path'],p['sha256'])) for p in pool['plans']]
    assert all(t['split']=='train' for p in plans for t in p['tasks'])
    assert len(pool['jobs'])==len({j['seed'] for j in pool['jobs']})
    source=Path(recipe['source']);seal=read(pin(source/'complete.json'))
    model=read(pin(source/'weights.json',seal['weights_sha256']));assert model['execution_mode']=='geometry_aim_explore_v1'
    assert model['geometry_aim_exploration']['offset_degrees']==OFFSETS
    members=[];requests=[];cases=[];histogram=collections.Counter();outcomes=collections.Counter();checked=0;excluded=collections.Counter()
    for job in sorted(pool['jobs'],key=lambda j:j['job']):
        binding=bindings[job['job']];capture=read(pin(Path(job['root'])/'report.json',binding['capture_sha256']))
        assert capture['capture_complete'] and capture['provenance_valid'] and len(capture['results'])==1
        result=capture['results'][0];assert result['capture_valid'] and result['dispatch_valid']
        manifest=read(pin(Path(job['root'])/'manifest.json'));log=pin(Path(result['root'])/'server.log',binding['server_log_sha256'])
        with log.open(encoding='utf-8-sig') as stream:generation,release,_=release_boundary(stream,manifest,job['seed'])
        steps=pin(Path(result['root'])/'dataset/steps.jsonl');windows={};seen=set();case_checked=0
        with steps.open(encoding='utf-8-sig') as stream:
            for step in map(json.loads,stream):
                o=step['observation'];n=step.get('native_step');d=step.get('server_execution') or {}
                if step['owner']!='provider' or o['identity']['life']!=1 or o['health']<=0 or o['identity']['frame']<release:continue
                if not n or not d.get('matched') or not d.get('window_exclusive') or d.get('recovery_commands')!=0:
                    excluded['unmatched_or_recovered_provider_frame']+=1;continue
                assert n['spawncount']==generation
                assert o['identity']['frame'] not in seen;seen.add(o['identity']['frame'])
                info=audit_sample(step,job['seed'])
                if info:
                    histogram[info['candidate']]+=1;checked+=1;case_checked+=1
                sample=step.get('sample');assert sample
                applied=step['applied_action'];proposed=step['action']
                if max(abs(applied[k]-proposed[k]) for k in ('yaw_delta_degrees','pitch_delta_degrees'))>.006:
                    excluded['applied_aim_changed_beyond_quantization']+=1;continue
                key=(n['spawncount'],n['begin_frame'],n['actor'],n['sequence'])
                assert key not in windows;windows[key]=(step,info)
        with log.open(encoding='utf-8-sig') as stream:shots=[s for s in scan(stream,set(windows)) if s['eligible']]
        task=plans[job['plan_index']]['tasks'][job['task_index']];episode=task['episode']['id']
        for shot in shots:
            step,info=windows[shot['window']];sample=step['sample'];label=outcome(shot);outcomes[label]+=1
            compact={k:sample[k] for k in ('latent','vertical','weapon','target','aim_mode') if k in sample}
            requests.append(dict(observation=step['observation'],sample=compact))
            members.append(dict(source=0,job=job['job'],seed=job['seed'],episode=episode,index=step['index'],
                frame=step['observation']['identity']['frame'],shot=int(shot['fields']['shot']),outcome=label,positive=label=='live_monster_damage',
                selected_target=sample.get('target',0)>0,seed_holdout=job['seed']%4==3,site_holdout='-site-02-' in episode,
                exploration_eligible=info is not None,candidate=info['candidate'] if info else -1,
                intervened=bool(info and info['intervened']),effective_intervention=bool(info and info['effective'])))
        cases.append(dict(job=job['job'],seed=job['seed'],episode=episode,shots=len(shots),sampler_checked_frames=case_checked))
    assert members and checked and any(m['intervened'] for m in members)
    assert any(m['exploration_eligible'] and not m['intervened'] for m in members)
    assert len({(m['seed'],m['shot']) for m in members})==len(members)
    args.out.mkdir(parents=True);input_path=args.out/'requests.jsonl';output=args.out/'native-features.jsonl'
    with input_path.open('w',encoding='utf-8') as f:
        for row in requests:f.write(json.dumps(row,separators=(',',':'))+'\n')
    pin(args.exporter)
    subprocess.run([str(args.exporter.resolve()),'--input',str(input_path.resolve()),'--out',str(output.resolve()),
        '--geometry-version','combat_observed_action_geometry_v2'],check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    with output.open(encoding='utf-8') as f:packed=[json.loads(line) for line in f]
    assert len(packed)==len(members)
    from ppo_combat import torch,training_devices
    training_devices();torch.set_default_device('cuda')
    from combat_action_geometry import features_for_version,names_for_version,SMOOTH_VERSION
    from process_combat_architecture_pool import compact_closed_exports
    x=torch.tensor([p['causal'] for p in packed],dtype=torch.float32,device='cuda')
    geometry=features_for_version(x,SMOOTH_VERSION)
    values=dict(version='combat_shot_prediction_data_v3',features=x[:,:181],features_with_recoil=x,action_geometry=geometry,
        geometry_version=SMOOTH_VERSION,geometry_names=list(names_for_version(SMOOTH_VERSION)),purpose='predictive_only',behavior_sampler=model['execution_mode'])
    values['labels']=torch.tensor([m['positive'] for m in members],dtype=torch.float32,device='cuda')
    for key in ('selected_target','seed_holdout','site_holdout','exploration_eligible','intervened','effective_intervention'):
        values[key]=torch.tensor([m[key] for m in members],dtype=torch.bool,device='cuda')
    assert torch.equal(values['selected_target'],x[:,151]==0)
    torch.save(values,args.out/'dataset.pt')
    with gzip.open(args.out/'members.jsonl.gz','wt',encoding='utf-8') as f:
        for m in members:f.write(json.dumps(m)+'\n')
    compact_closed_exports(args.out)
    for path,digest in pins.items():assert sha(Path(path))==digest,path
    report=dict(state='complete',version='combat_predictive_exploration_dataset_v1',device='cuda',purpose='shot_predictive_train_only',
        behavior_sampler=model['execution_mode'],source_cases=len(cases),cases=cases,shots=len(members),
        battles_with_shots=sum(c['shots']>0 for c in cases),outcomes=dict(outcomes),sampler_checked_frames=checked,
        assignment_histogram=dict(histogram),perturbed_shots=sum(m['intervened'] for m in members),
        effective_perturbed_shots=sum(m['effective_intervention'] for m in members),
        eligible_baseline_shots=sum(m['exploration_eligible'] and not m['intervened'] for m in members),
        excluded=dict(excluded),inputs=pins,recipe_sha256=sha(root/'recipe.json'),script_sha256=sha(__file__),
        dataset_sha256=sha(args.out/'dataset.pt'),members_sha256=sha(args.out/'members.jsonl.gz'),
        request_sha256=sha(input_path),feature_export_sha256=sha(output),ppo_ready=False,policy_training=False,promotion=False,
        scope='Fresh TRAIN actual MG outcomes under randomized bounded aim assignments. Native truth labels only; causal observation/action packing in pure Go and geometry on CUDA. No log_probability/value/PPO targets exported, no neural Go replay or fitting. Folds are metadata; small pilots may not span both seed sides.')
    save(args.out/'report.json',report)
    save(args.out/'complete.json',dict(state='complete',report_sha256=sha(args.out/'report.json'),dataset_sha256=report['dataset_sha256'],ppo_ready=False,policy_training=False))
    print(json.dumps({k:report[k] for k in ('source_cases','shots','sampler_checked_frames','perturbed_shots','effective_perturbed_shots','eligible_baseline_shots','outcomes')}))


if __name__=='__main__':main()
