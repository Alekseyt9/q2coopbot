"""Separate Gaussian aiming noise from mean error on frozen TRAIN shot poses.

Read-only CUDA policy replay. Counterfactual rays are geometric diagnostics,
not executed actions, predicted hits, PPO data, or estimates of battle success.
"""
import argparse
import collections
import json
import math
from pathlib import Path
from evaluate_registered_combat_models import read,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','corpus','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();assert not args.out.exists()
    source,corpus,out=[p.resolve() for p in (args.source,args.corpus,args.out)]
    pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path)
        assert expected is None or digest==expected,str(path)
        pins[str(path)]=digest;return path
    seal=read(pin(source/'complete.json'))
    for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:pin(source/name,seal[key])
    model=read(source/'weights.json');assert not model['deterministic'] and not model.get('execution_mode')
    data=corpus/'processing-train/merged';meta=read(pin(data/'report.json'))
    assert meta['model_sha256']==seal['weights_sha256']
    pin(data/'rollout.jsonl',meta['rollout_sha256']);pin(data/'sequence.jsonl',meta['sequence_sha256'])
    closed=read(pin(corpus/'complete.json'));assert closed['state']=='complete'
    pool=read(pin(corpus/'pool/report.json',closed['pool_sha256']))
    assert pool['state']=='complete' and pool['source_unchanged'] and not any(j['error'] for j in pool['jobs'])
    plans=[read(pin(p['path'],p['sha256'])) for p in pool['plans']]
    assert all(t['split']=='train' for p in plans for t in p['tasks'])
    out.mkdir(parents=True)
    protocol=dict(version='combat_aim_noise_decomposition_v1',scales=[1.,.25,0.],
        fixed='TRAIN histories, selected target/mode, sampled movement/pose/weapon, shot muzzle, observed target point and actual recoil',
        alignment_max_degrees=.05,executed_delta_quantization_max_degrees=.006,
        baseline_ray_formatting_max_degrees=.01,
        maximum_native_log_probability_error=.003,
        scope='One-step geometric rays at existing shots only; no visibility, hit, subsequent trajectory or win prediction. No new neural execution on CPU; no fitting or policy promotion.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    from ppo_combat import torch,training_devices
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    from ppo_recurrent import prepare_context,sequence,durable_json
    from migrate_combat_navigation_cuda import branch
    from fork_conditional_weapon_exploration_cuda import equal_tree
    from combat_precision_head import selected_means,physical_actions,probabilities
    from native_combat_release import release_boundary
    from report_combat_machinegun_hits import scan,vector
    from report_combat_machinegun_aim import measure,angular,ray
    for name in ('combat_precision_head.py','combat_sampled_motion_aim.py','report_combat_machinegun_aim.py','report_combat_machinegun_hits.py','native_projectile_window.py'):
        pin(Path(__file__).with_name(name))
    cp=torch.load(source/'checkpoint.pt',map_location='cuda',weights_only=True)
    actor=branch(model,'actor');assert equal_tree(actor.state_dict(),cp['actor'])
    with (data/'rollout.jsonl').open(encoding='utf-8-sig') as f:rows=list(map(json.loads,f))
    with (data/'sequence.jsonl').open(encoding='utf-8-sig') as f:context=list(map(json.loads,f))
    prepared=prepare_context(context,rows,'cuda')
    features=torch.tensor([r['features'] for r in rows],dtype=torch.float32,device='cuda')
    latent=torch.tensor([r['sample']['latent'] for r in rows],dtype=torch.float32,device='cuda')
    category=lambda name:torch.tensor([r['sample'].get(name,0) for r in rows],dtype=torch.long,device='cuda')
    target,mode,vertical,weapon=(category(k) for k in ('target','aim_mode','vertical','weapon'))
    with torch.no_grad():
        raw,memory_error=sequence(actor,prepared,check_states=True)
        means=selected_means(raw,target,mode,features,latent[:,:2],actor.sampled_motion_aim_branch,vertical,actor.motor_branch)
        attack=torch.tensor([float(r['sample']['attack']) for r in rows],device='cuda')
        lp=probabilities(raw,torch.tensor(model['log_std'],device='cuda'),latent,attack,vertical,weapon,features,target,mode,
            actor.sampled_motion_aim_branch,actor.motor_branch,actor.conditional_fire_branch)[0]
        lp_error=float((lp-torch.tensor([r['sample']['log_probability'] for r in rows],device='cuda')).abs().max())
        assert lp_error<protocol['maximum_native_log_probability_error']
        actions={}
        for scale in protocol['scales']:
            z=latent.clone();z[:,2:4]=means[:,2:4]+scale*(latent[:,2:4]-means[:,2:4])
            if scale==1.:z=latent.clone()
            assert torch.equal(z[:,:2],latent[:,:2])
            actions[scale]=physical_actions(z,mode).cpu().tolist()
    lookup={(r['seed'],r['index'],r['frame']):i for i,r in enumerate(rows)}
    assert len(lookup)==len(rows)
    shots_out=[];excluded=collections.Counter();all_statuses=collections.Counter();case_rows=[];baseline_error=0.
    for job in sorted(pool['jobs'],key=lambda j:j['job']):
        task=plans[job['plan_index']]['tasks'][job['task_index']]
        capture=read(pin(Path(job['root'])/'report.json'))
        assert capture['capture_complete'] and capture['provenance_valid'] and len(capture['results'])==1
        result=capture['results'][0];assert result['capture_valid'] and result['dispatch_valid']
        manifest=read(pin(Path(job['root'])/'manifest.json'))
        log=pin(Path(result['root'])/'server.log')
        with log.open(encoding='utf-8-sig') as f:generation,release,game_release=release_boundary(f,manifest,job['seed'])
        steps=pin(corpus/'processing-train'/f"case-{job['job']:04d}/native/replay-0/steps.jsonl")
        windows={}
        with steps.open(encoding='utf-8-sig') as f:
            for s in map(json.loads,f):
                o=s['observation'];n=s.get('native_step');d=s.get('server_execution') or {}
                if not n or s['owner']!='provider' or o['identity']['life']!=1 or o['health']<=0 or o['identity']['frame']<release:continue
                if not d.get('matched') or not d.get('window_exclusive') or d.get('recovery_commands')!=0:continue
                assert n['spawncount']==generation
                index=lookup.get((job['seed'],s['index'],o['identity']['frame']))
                if index is None:continue
                assert s['sample']['latent']==rows[index]['sample']['latent']
                key=(n['spawncount'],n['begin_frame'],n['actor'],n['sequence'])
                assert key not in windows;windows[key]=(s,index)
        with log.open(encoding='utf-8-sig') as f:shots=[s for s in scan(f,set(windows)) if s['eligible']]
        accepted=0
        for shot in shots:
            s,index=windows[shot['window']];m=measure(shot,s);all_statuses[m['status']]+=1
            if m['status']!='measured':excluded[m['status']]+=1;continue
            actual=actions[1.][index];a=s['action'];applied=s['applied_action']
            assert max(abs(actual[i]-a[name]) for i,name in ((2,'yaw_delta_degrees'),(3,'pitch_delta_degrees')))<.001
            if max(abs(a[k]-applied[k]) for k in ('yaw_delta_degrees','pitch_delta_degrees'))>protocol['executed_delta_quantization_max_degrees']:
                excluded['changed_executed_aim']+=1;continue
            if max(m['native_vs_applied_pitch'],m['native_vs_applied_yaw'])>protocol['alignment_max_degrees']:
                excluded['unaligned_native_view']+=1;continue
            o=s['observation'];enemy=next(e for e in o['enemies'] if e['id']==a['target_entity'] and e.get('observed_track',0)==a.get('target_track',0))
            solid=enemy['observed_solid'];bottom=-((solid>>5)&31)*8;top=((solid>>10)&63)*8-32
            height=(top+bottom)/2 if top-bottom<16 else min(max(22,bottom+8),top-8)
            start=vector(shot['fields']['start']);view=vector(shot['fields']['view']);kick=vector(shot['fields']['recoil'])
            desired=[o['position'][axis]+enemy['relative'][axis]-start[axis] for axis in range(3)];desired[2]+=height
            errors={};unsupported=False
            for scale,values in actions.items():
                yaw=view[1]+values[index][2]-actual[2]
                pitch=view[0]+values[index][3]-actual[3]
                if not -89<=pitch<=89:unsupported=True;break
                errors[str(scale)]=angular(ray(yaw+kick[1],pitch+kick[0]),desired)
            if unsupported:excluded['counterfactual_pitch_outside_execution_range']+=1;continue
            # Native aim/view/recoil use %.6g telemetry. Compare at a physical
            # angular tolerance instead of requiring decimal-format identity.
            discrepancy=abs(errors['1.0']-m['post_recoil_degrees'])
            baseline_error=max(baseline_error,discrepancy)
            assert discrepancy<protocol['baseline_ray_formatting_max_degrees'],discrepancy
            shots_out.append(dict(seed=job['seed'],job=job['job'],episode=task['episode']['id'],frame=o['identity']['frame'],
                shot=int(shot['fields']['shot']),mode=s['sample'].get('aim_mode',0),distance=m['distance'],
                moving=m['native_horizontal_speed']>=10,strong_recoil=abs(m['native_recoil_pitch'])>6,
                selected_target_damage=m['selected_target_damage'],errors=errors))
            accepted+=1
        case_rows.append(dict(job=job['job'],seed=job['seed'],eligible_shots=len(shots),diagnostic_shots=accepted))
    def summarize(selected):
        result=dict(shots=len(selected),battles=len({s['seed'] for s in selected}),scales={})
        for scale in protocol['scales']:
            values=sorted(s['errors'][str(scale)] for s in selected)
            result['scales'][str(scale)]=dict(mean_degrees=sum(values)/len(values) if values else None,
                median_degrees=values[len(values)//2] if values else None,
                within_2_degrees=sum(v<=2 for v in values),within_5_degrees=sum(v<=5 for v in values),
                better_than_sampled=sum(s['errors'][str(scale)]<s['errors']['1.0'] for s in selected))
        return result
    groups={'all':shots_out,'fine':[s for s in shots_out if s['mode']==1],'coarse':[s for s in shots_out if s['mode']==0]}
    for name,predicate in [('ordinary',lambda s:not s['episode'].endswith('-recovery')),('recovery',lambda s:s['episode'].endswith('-recovery')),
            ('moving',lambda s:s['moving']),('stationary',lambda s:not s['moving']),('near',lambda s:s['distance']<256),
            ('medium',lambda s:256<=s['distance']<512),('far',lambda s:s['distance']>=512),('strong_recoil',lambda s:s['strong_recoil'])]:
        groups[name]=[s for s in shots_out if predicate(s)]
    by_battle=[dict(seed=seed,**summarize([s for s in shots_out if s['seed']==seed])) for seed in sorted({s['seed'] for s in shots_out})]
    report=dict(state='complete',version=protocol['version'],device='cuda',rows=len(rows),memory_max_error=memory_error,
        native_log_probability_max_error=lp_error,eligible_shots=sum(c['eligible_shots'] for c in case_rows),
        native_statuses=dict(all_statuses),excluded=dict(excluded),groups={k:summarize(v) for k,v in groups.items()},
        baseline_ray_formatting_max_error_degrees=baseline_error,
        cases=case_rows,by_battle=by_battle,inputs=pins,protocol_sha256=sha(out/'protocol.json'),script_sha256=sha(__file__),
        training_performed=False,promotion=False,scope=protocol['scope'])
    for path,digest in pins.items():assert sha(Path(path))==digest,path
    durable_json(out/'report.json',report)
    durable_json(out/'complete.json',dict(state='complete',report_sha256=sha(out/'report.json'),training_performed=False,promotion=False))
    print(json.dumps(dict(eligible_shots=report['eligible_shots'],excluded=report['excluded'],groups=report['groups'])))


if __name__=='__main__':main()
