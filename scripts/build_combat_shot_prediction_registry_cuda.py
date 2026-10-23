"""Build a compact causal shot dataset from verified TRAIN corpus recipes."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:return [json.loads(s) for s in stream]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--registry',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--action-geometry',action='store_true',help='Add causal recoil4 and action-relative geometry29; keep legacy181 intact')
    args=ap.parse_args();assert not args.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path);assert expected is None or digest==expected,str(path)
        pins[str(path)]=digest;return path
    registry=read(pin(args.registry));assert registry['version']=='combat_shot_prediction_source_registry_v1' and registry['state']=='sources_verified'
    assert len(registry['entries'])==len({v['corpus'] for v in registry['entries']})>0
    from report_combat_machinegun_hits import scan
    from report_registered_machinegun_outcomes import outcome
    args.out.mkdir(parents=True)
    version='combat_shot_prediction_dataset_v3' if args.action_geometry else 'combat_shot_prediction_dataset_v2'
    save(args.out/'protocol.json',dict(version=version,registry_sha256=sha(args.registry),
        input=('fire181_plus_recoil4_plus_action_geometry29' if args.action_geometry else 'conditional_fire181_causal_inputs'),label='immediate_native_MG_live_monster_damage',
        duplicate_rule='same_episode_seed_and_entire_ordered_shot_features_actions_labels',
        seed_fold='seed_mod4_eq3_grouped_by_episode_seed_across_sources',geometry_fold='all_site02',
        policy_training=False,ppo_ready=False,promotion=False))
    members=[];features=[];samples=[];cases=[];signatures={};groups={}
    for source_index,entry in enumerate(registry['entries']):
        root=Path(entry['corpus']);closed=read(pin(root/'complete.json',entry['complete_sha256']))
        assert closed['state']=='complete' and not closed['training_performed']
        protocol=read(pin(root/'protocol.json',closed['protocol_sha256']));assert protocol['train_only']
        pool=read(pin(root/'pool/report.json',entry['pool_sha256']))
        assert pool['state']=='complete' and pool['source_unchanged'] and len(pool['plans'])==1
        assert not any(j['error'] for j in pool['jobs'])
        binding=pool['plans'][0];plan=read(pin(binding['path'],binding['sha256']))
        assert all(t['split']=='train' and t['modes']==['learned'] for t in plan['tasks'])
        model=read(pin(plan['model_path'],entry['model_sha256']))
        assert model['feature_version']=='combat_features_v10' and model.get('aim_mode_head')=='combat_target_coarse_fine_aim_v1'
        meta=read(pin(root/'processing-train/merged/report.json',entry['merged_report_sha256']))
        assert meta['model_sha256']==entry['model_sha256'] and meta['rows']==entry['eligible_rows']
        observed=rows(pin(root/'processing-train/merged/rollout.jsonl',entry['rollout_sha256']))
        lookup={(v['seed'],v['index']):v for v in observed};assert len(lookup)==len(observed)==meta['rows']
        for job in sorted(pool['jobs'],key=lambda v:v['job']):
            assert job['mode']=='learned';task=plan['tasks'][job['task_index']]
            assert job['seed'] in task['seeds']
            episode=task['episode']['id'];group=(episode,job['seed'])
            folds=(job['seed']%4==3,'-site-02-' in episode)
            assert group not in groups or groups[group]==folds;groups[group]=folds
            capture=read(pin(Path(job['root'])/'report.json'))
            assert capture['capture_complete'] and capture['provenance_valid'] and len(capture['results'])==1
            native=root/f"processing-train/case-{job['job']:04d}/native/replay-0"
            windows={}
            for step in rows(pin(native/'steps.jsonl')):
                if (job['seed'],step['index']) not in lookup:continue
                dispatch=step.get('server_execution') or {};n=step.get('native_step')
                if not n or step['owner']!='provider' or not dispatch.get('matched') or not dispatch.get('window_exclusive') or dispatch.get('recovery_commands',0):continue
                key=(n['spawncount'],n['begin_frame'],n['actor'],n['sequence']);assert key not in windows
                windows[key]=step
            log=pin(Path(capture['results'][0]['root'])/'server.log')
            with log.open(encoding='utf-8-sig') as stream:shots=[s for s in scan(stream,set(windows)) if s['eligible']]
            candidate=[];signature=hashlib.sha256()
            for shot in shots:
                step=windows[shot['window']];row=lookup[(job['seed'],step['index'])];sample=row['sample']
                assert sample['attack'] and row['frame']==step['observation']['identity']['frame']
                result=outcome(shot)
                record=dict(source=source_index,job=job['job'],seed=job['seed'],episode=episode,index=row['index'],frame=row['frame'],
                    shot=int(shot['fields']['shot']),outcome=result,positive=result=='live_monster_damage',selected_target=sample.get('target',0)>0,
                    seed_holdout=folds[0],site_holdout=folds[1])
                action=[sample.get(k,0) for k in ('target','aim_mode','vertical','weapon')]+sample['latent']
                # Ignore model version, neural memory and actor IDs; require exact
                # observed features/actions/labels throughout the battle instead.
                signature.update(json.dumps([row['frame'],row['features'],action,result],separators=(',',':'),allow_nan=False).encode())
                candidate.append((record,row['features'],sample))
            key=(*group,signature.hexdigest())
            duplicate=signatures.get(key) if candidate else None
            case=dict(source=source_index,job=job['job'],seed=job['seed'],episode=episode,shots=len(candidate),trajectory_sha256=signature.hexdigest(),duplicate_of=duplicate)
            cases.append(case)
            if duplicate is not None:continue
            if candidate:signatures[key]=dict(source=source_index,job=job['job'])
            for record,values,sample in candidate:members.append(record);features.append(values);samples.append(sample)
        save(args.out/'progress.json',dict(stage='extracting',sources_done=source_index+1,sources_total=len(registry['entries']),shots=len(members)))
        print(json.dumps(dict(source=source_index,shots=len(members))),flush=True)
    assert len(cases)==registry['native_cases'] and members
    assert len({(v['source'],v['seed'],v['shot']) for v in members})==len(members)
    for field in ('seed_holdout','site_holdout'):
        before={(v['episode'],v['seed']) for v in members if not v[field]}
        after={(v['episode'],v['seed']) for v in members if v[field]}
        assert before and after and before.isdisjoint(after)
    with gzip.open(args.out/'members.jsonl.gz','wt',encoding='utf-8') as stream:
        for member in members:stream.write(json.dumps(member)+'\n')
    from ppo_combat import torch,training_devices
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    from combat_conditional_fire import inputs
    x=torch.tensor(features,dtype=torch.float32);latent=torch.tensor([s['latent'] for s in samples],dtype=torch.float32)
    cat=lambda key:torch.tensor([s.get(key,0) for s in samples],dtype=torch.long)
    values=inputs(x,cat('target'),cat('aim_mode'),latent,cat('vertical'),cat('weapon'))
    tensors=dict(version='combat_shot_prediction_data_v2',features=values,
        labels=torch.tensor([v['positive'] for v in members],dtype=torch.float32),
        selected_target=torch.tensor([v['selected_target'] for v in members],dtype=torch.bool),
        seed_holdout=torch.tensor([v['seed_holdout'] for v in members],dtype=torch.bool),
        site_holdout=torch.tensor([v['site_holdout'] for v in members],dtype=torch.bool))
    if args.action_geometry:
        from combat_action_geometry import features as geometry,NAMES,VERSION
        causal=torch.cat((values,x[:,810:814]),-1)
        tensors.update(version='combat_shot_prediction_data_v3',features_with_recoil=causal,action_geometry=geometry(causal),
            geometry_version=VERSION,geometry_names=list(NAMES))
        pins[str(Path(__file__).with_name('combat_action_geometry.py').resolve())]=sha(Path(__file__).with_name('combat_action_geometry.py'))
    torch.save(tensors,args.out/'dataset.pt')
    for path,digest in pins.items():assert sha(Path(path))==digest,'Changed input: '+path
    report=dict(state='complete',version=version,device='cuda',sources=len(registry['entries']),source_cases=len(cases),
        battles_with_shots=sum(c['shots']>0 for c in cases),empty_battles=sum(c['shots']==0 for c in cases),
        duplicate_battles=sum(c['duplicate_of'] is not None for c in cases),shots=len(members),
        outcomes=dict(collections.Counter(v['outcome'] for v in members)),cases=cases,inputs=pins,script_sha256=sha(__file__),
        dataset_sha256=sha(args.out/'dataset.pt'),members_sha256=sha(args.out/'members.jsonl.gz'),
        policy_training=False,ppo_ready=False,promotion=False,
        scope='Verified existing TRAIN-only supervision registry. Whole-battle duplicate removal, group-safe seed/geometry folds. Native outcomes are labels only. Off-policy predictive dataset, never a merged PPO rollout.')
    save(args.out/'report.json',report)
    save(args.out/'complete.json',dict(state='complete',report_sha256=sha(args.out/'report.json'),dataset_sha256=report['dataset_sha256'],policy_training=False,ppo_ready=False))
    save(args.out/'progress.json',dict(stage='complete',shots=len(members)))
    print(json.dumps({k:report[k] for k in ('sources','source_cases','battles_with_shots','empty_battles','duplicate_battles','shots','outcomes')}))


if __name__=='__main__':main()
