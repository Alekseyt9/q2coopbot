"""Recheck live learned aim rankings on CUDA; never standard PPO replay."""
import argparse
import copy
import itertools
import json
import subprocess
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation','spec','exporter','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--label',default='ranker')
    args=ap.parse_args();assert not args.out.exists();args.out.mkdir(parents=True);pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path);assert expected is None or digest==expected,str(path)
        pins[str(path)]=digest;return path
    root=args.evaluation;closed=read(pin(root/'complete.json'));assert closed['state']=='complete'
    spec=read(pin(args.spec,closed['spec_sha256']));labels=[v['label'] for v in spec['variants']];index=labels.index(args.label)
    variant=spec['variants'][index];assert not variant['cuda_replay']
    seal=read(pin(Path(variant['update'])/'complete.json'));model=read(pin(Path(variant['update'])/'weights.json',seal['weights_sha256']))
    assert model['execution_mode']=='geometry_shot_rank_aim_v1'
    ranker=model['geometry_aim_ranker'];predictor=ranker['predictor'];offsets=list(itertools.product(ranker['offset_degrees'],repeat=2))
    assert len(offsets)==25 and offsets[0]==(0,0)
    pool=read(pin(root/'pool/report.json',closed['pool_sha256']))
    execution=read(pin(root/('execution-'+args.label+'.json')));assert execution['pool_sha256']==closed['pool_sha256']
    bindings={v['job']:v for v in execution['cases']};members=[];requests=[];cases=[]
    for job in sorted(pool['jobs'],key=lambda j:j['job']):
        if job['plan_index']!=index:continue
        binding=bindings[job['job']]
        capture=read(pin(Path(job['root'])/'report.json',binding['capture_sha256']));result=capture['results'][0]
        assert capture['capture_complete'] and capture['provenance_valid'] and result['capture_valid'] and result['dispatch_valid']
        steps=pin(Path(result['root'])/'dataset/steps.jsonl');count=0
        with steps.open(encoding='utf-8-sig') as stream:
            for step in map(json.loads,stream):
                o=step['observation'];d=step.get('server_execution') or {};n=step.get('native_step');s=step.get('sample') or {};r=s.get('aim_ranking')
                if not r:continue
                if step['owner']!='provider' or o['identity']['life']!=1 or o['health']<=0 or o['identity']['frame']<binding['network_release_frame']:continue
                if not n or not d.get('matched') or not d.get('window_exclusive') or d.get('recovery_commands')!=0:continue
                assert n['spawncount']==binding['release_generation'] and r['candidates']==25 and 0<=r['candidate']<25
                assert o['weapon'] in ('Machinegun','models/weapons/v_machn/tris.md2')
                before=copy.deepcopy(s);before['latent'][2:4]=r['source_latent'];before.pop('aim_ranking')
                requests.append(dict(observation=o,sample=before));members.append(dict(job=job['job'],sample=s,ranking=r));count+=1
        cases.append(dict(job=job['job'],ranked_frames=count))
    assert members,'Ranker never activated'
    input_path=args.out/'requests.jsonl';packed=args.out/'native-features.jsonl'
    with input_path.open('w',encoding='utf-8') as f:
        for row in requests:f.write(json.dumps(row,separators=(',',':'))+'\n')
    pin(args.exporter)
    motor=predictor.get('motor')
    pack_version='combat_observed_action_geometry_v3' if motor else predictor['geometry_version']
    subprocess.run([str(args.exporter.resolve()),'--input',str(input_path.resolve()),'--out',str(packed.resolve()),
        '--geometry-version',pack_version],check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    from ppo_combat import torch,training_devices
    training_devices();torch.set_default_device('cuda')
    from combat_action_geometry import features_for_version
    from process_combat_architecture_pool import compact_closed_exports
    pin(Path(__file__).with_name('combat_action_geometry.py'))
    with packed.open(encoding='utf-8') as f:rows=[json.loads(line) for line in f]
    base=torch.tensor([r['causal'] for r in rows],dtype=torch.float64,device='cuda')
    assert len(base)==len(members)
    limit=torch.where(base[:,161]>0,15.,180.)
    candidates=base[:,None,:].expand(-1,25,-1).clone()
    grid=torch.tensor(offsets,dtype=torch.float64,device='cuda')
    ratio=((base[:,None,164:166]*180+grid[None])/limit[:,None,None]).clamp(-1+1e-6,1-1e-6)
    candidates[:,1:,164:166]=torch.tanh(torch.atanh(ratio[:,1:]))*limit[:,None,None]/180
    flat=candidates.reshape(-1,185)
    if motor:
        from combat_action_effect import inputs,FactorizedMotor,projected_geometry
        pin(Path(__file__).with_name('combat_action_effect.py'))
        time=torch.tensor([[r['observation']['observation_age_ms']/300,r['observation']['previous_applied_command']['Msec']/100] for r in requests],dtype=torch.float64,device='cuda')
        time=time[:,None,:].expand(-1,25,-1).reshape(-1,2)
        net=FactorizedMotor().double()
        with torch.no_grad():
            for branch,name in ((net.position,'position_layers'),(net.direction,'direction_layers')):
                for i,l in zip((0,2),motor[name]):
                    branch[i].weight.copy_(torch.tensor(l['weight'],dtype=torch.float64,device='cuda'));branch[i].bias.copy_(torch.tensor(l['bias'],dtype=torch.float64,device='cuda'))
            values=inputs(flat,time)
            forecast=net((values-torch.tensor(motor['mean'],dtype=torch.float64,device='cuda'))/torch.tensor(motor['std'],dtype=torch.float64,device='cuda'))
            geo=projected_geometry(flat,forecast,False)
    else:geo=features_for_version(flat,predictor['geometry_version'])
    assert bool((geo[:,0]==1).all())
    x=torch.cat((flat,geo),-1)
    # Float64 CUDA mirrors Go's live arithmetic on the exported FP32 weights.
    mean=torch.tensor(predictor['mean'],dtype=torch.float64,device='cuda');std=torch.tensor(predictor['std'],dtype=torch.float64,device='cuda')
    weight=torch.tensor(predictor['weight'],dtype=torch.float64,device='cuda')
    logits=(((x-mean)/std)@weight+predictor['bias']).reshape(-1,25)
    choices=torch.tensor([m['ranking']['candidate'] for m in members],dtype=torch.long,device='cuda')
    selected=logits.gather(1,choices[:,None])[:,0]
    before=torch.tensor([m['ranking']['before_logit'] for m in members],dtype=torch.float64,device='cuda')
    after=torch.tensor([m['ranking']['after_logit'] for m in members],dtype=torch.float64,device='cuda')
    score_error=float(torch.maximum((before-logits[:,0]).abs(),(after-selected).abs()).max())
    gap=float((logits.amax(-1)-selected).max())
    assert score_error<1e-6,score_error
    assert gap<1e-6,gap
    actual=torch.tensor([m['sample']['latent'][2:4] for m in members],dtype=torch.float64,device='cuda').tanh()*limit[:,None]
    expected=candidates.gather(1,choices[:,None,None].expand(-1,1,185))[:,0,164:166]*180
    angular_error=float((actual-expected).abs().max());assert angular_error<1e-5,angular_error
    changed=int((choices!=0).sum());assert changed>0,'No nonbaseline candidate selected'
    compact_closed_exports(args.out)
    for path,digest in pins.items():assert sha(Path(path))==digest,path
    report=dict(state='complete',version='combat_geometry_aim_ranker_live_cuda_audit_v1',device='cuda',ranked_frames=len(members),
        changed_frames=changed,cases=cases,candidate_score_max_error=score_error,selected_score_gap=gap,
        selected_angle_max_error_degrees=angular_error,inputs=pins,script_sha256=sha(__file__),
        standard_ppo_replay=False,training_performed=False,promotion=False,
        scope='Actual native ranked observations/actions. Pure Go feature packing only offline; all learned score rechecks on CUDA. Confirms live choices against all25candidates, not quality or validity of predicted counterfactual hits.')
    save(args.out/'report.json',report);save(args.out/'complete.json',dict(state='complete',report_sha256=sha(args.out/'report.json')))
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','scope','cases')}))


if __name__=='__main__':main()
