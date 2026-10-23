"""Held-out native motor error and actual-shot calibration; CUDA, no fitting."""
import argparse
import collections
import gzip
import json
import math
import subprocess
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from report_combat_machinegun_hits import scan,vector
from report_registered_machinegun_outcomes import outcome


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('evaluation','spec','exporter','out'):ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();a.out.mkdir(parents=True);pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or h==expected,str(path)
        pins[str(path)]=h;return path
    root=a.evaluation.resolve();closed=read(pin(root/'complete.json'))
    assert closed['state']=='complete' and not closed['training_performed']
    spec=read(pin(a.spec,closed['spec_sha256']));pool=read(pin(root/'pool/report.json',closed['pool_sha256']))
    assert pool['state']=='complete' and pool['source_unchanged']
    mg=read(pin(root/'mg-outcomes.json'));assert mg['state']=='complete'
    predictors={}
    for label in ('old-ranker','motor-ranker'):
        v=next(v for v in spec['variants'] if v['label']==label)
        source=Path(v['update']);seal=read(pin(source/'complete.json'))
        model=read(pin(source/'weights.json',seal['weights_sha256']))
        predictors[label]=model['geometry_aim_ranker']['predictor']
    assert predictors['motor-ranker']['geometry_version']=='combat_predicted_action_geometry_v1'
    requests=[];members=[];case_counts=[]
    for index,v in enumerate(spec['variants']):
        if v['mode']!='learned':continue
        label=v['label'];execution=read(pin(root/('execution-'+label+'.json')))
        assert execution['pool_sha256']==closed['pool_sha256'] and execution['selected_plan_index']==index
        bindings={c['job']:c for c in execution['cases']}
        for job in sorted(pool['jobs'],key=lambda j:j['job']):
            if job['plan_index']!=index:continue
            b=bindings[job['job']];capture=read(pin(Path(job['root'])/'report.json',b['capture_sha256']))
            assert capture['capture_complete'] and capture['provenance_valid']
            raw=Path(capture['results'][0]['root'])
            log=pin(raw/'server.log',b['server_log_sha256']);steps=pin(raw/'dataset/steps.jsonl')
            windows={}
            with steps.open(encoding='utf-8-sig') as f:
                for s in map(json.loads,f):
                    o=s['observation'];n=s.get('native_step');d=s.get('server_execution') or {}
                    if not n or s['owner']!='provider' or o['identity']['life']!=1 or o['health']<=0 or o['identity']['frame']<b['network_release_frame']:continue
                    if not d.get('matched') or not d.get('window_exclusive') or d.get('recovery_commands')!=0:continue
                    assert n['spawncount']==b['release_generation']
                    key=tuple(n[k] for k in ('spawncount','begin_frame','actor','sequence'))
                    assert key not in windows;windows[key]=s
            with log.open(encoding='utf-8-sig') as f:shots=[s for s in scan(f,set(windows)) if s['eligible']]
            for shot in shots:
                s=windows[shot['window']];o=s['observation'];sample=s['sample'];action=s['action'];applied=s['applied_action']
                compact={k:sample[k] for k in ('latent','vertical','weapon','target','aim_mode') if k in sample}
                requests.append(dict(observation=o,sample=compact))
                r=sample.get('aim_ranking');changed_aim=max(abs(applied[k]-action[k]) for k in ('yaw_delta_degrees','pitch_delta_degrees'))>.006
                changed_move=max(abs(applied[k]-action[k]) for k in ('forward','side'))>.004
                members.append(dict(label=label,job=job['job'],episode=b['episode'],seed=job['seed'],index=s['index'],
                    shot=int(shot['fields']['shot']),outcome=outcome(shot),ranked=r is not None,
                    logged_after_logit=r['after_logit'] if r else None,logged_before_logit=r['before_logit'] if r else None,
                    observation_age_ms=o['observation_age_ms'],observed_mg=o['weapon'] in ('Machinegun','models/weapons/v_machn/tris.md2'),
                    applied_aim_changed=changed_aim,applied_motion_changed=changed_move,
                    native_muzzle_label=vector(shot['fields']['start']),native_aim_label=vector(shot['fields']['aim'])))
            case_counts.append(dict(label=label,job=job['job'],shots=len(shots)))
        assert sum(c['shots'] for c in case_counts if c['label']==label)==mg['variants'][label]['all']['shots']
    ip,op=a.out/'requests.jsonl',a.out/'pure-go-inputs.jsonl'
    with ip.open('w',encoding='utf-8') as f:
        for q in requests:f.write(json.dumps(q,separators=(',',':'))+'\n')
    pin(a.exporter)
    subprocess.run([str(a.exporter.resolve()),'--input',str(ip.resolve()),'--out',str(op.resolve()),
        '--geometry-version','combat_observed_action_geometry_v3','--motor-inputs'],check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    from ppo_combat import torch,training_devices
    from combat_action_effect import FactorizedMotor,projected_geometry
    from combat_action_geometry import features_for_version
    from process_combat_architecture_pool import compact_closed_exports
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for name in ('combat_action_effect.py','combat_action_geometry.py','report_combat_machinegun_hits.py','report_registered_machinegun_outcomes.py'):pin(Path(__file__).with_name(name))
    with op.open(encoding='utf-8') as f:packed=[json.loads(line) for line in f]
    assert len(packed)==len(members)
    tensor=lambda values:torch.tensor(values,dtype=torch.float64,device='cuda')
    c=tensor([r['causal'] for r in packed]);x=tensor([r['motor_inputs'] for r in packed]);motor=predictors['motor-ranker']['motor']
    net=FactorizedMotor().double()
    with torch.no_grad():
        for branch,key in ((net.position,'position_layers'),(net.direction,'direction_layers')):
            for i,l in zip((0,2),motor[key]):
                branch[i].weight.copy_(tensor(l['weight']));branch[i].bias.copy_(tensor(l['bias']))
        forecast=net((x-tensor(motor['mean']))/tensor(motor['std']))
        geometry=projected_geometry(c,forecast,False)
    probabilities={};logits={}
    for label,p in predictors.items():
        g=geometry if label=='motor-ranker' else features_for_version(c,p['geometry_version'])
        values=torch.cat((c,g),-1)
        logits[label]=((values-tensor(p['mean']))/tensor(p['std']))@tensor(p['weight'])+p['bias']
        probabilities[label]=logits[label].sigmoid()
    y=tensor([m['outcome']=='live_monster_damage' for m in members]);native=tensor([m['native_aim_label'] for m in members])
    pos=tensor([q['observation']['position'] for q in requests]);actual=tensor([m['native_muzzle_label'] for m in members])
    center=pos.clone();center[:,2]+=torch.where(c[:,3]>0,-10.,14.)
    yaw=torch.atan2(c[:,6],c[:,7]);cy,sy=yaw.cos(),yaw.sin()
    delta=forecast[:,:3]*32
    world=torch.stack((cy*delta[:,0]-sy*delta[:,1],sy*delta[:,0]+cy*delta[:,1],delta[:,2]),-1)
    muzzle_errors={'current_centered':(center-actual).norm(dim=-1),
                  'constant_velocity_100ms':(center+c[:,10:13]*400*.1-actual).norm(dim=-1),
                  'learned_motor':(center+world-actual).norm(dim=-1)}
    kick=torch.where(c[:,181:182]>0,c[:,182:185]*32,0.)
    angles=torch.stack((torch.atan2(c[:,8],c[:,9])*180/math.pi+c[:,165]*180+kick[:,0],yaw*180/math.pi+c[:,164]*180+kick[:,1]),-1)
    def direction(v):
        pitch,yaw=v[:,0]*math.pi/180,v[:,1]*math.pi/180
        return torch.stack((pitch.cos()*yaw.cos(),pitch.cos()*yaw.sin(),-pitch.sin()),-1)
    def angle(v):return torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(v,native),dim=-1),(v*native).sum(-1))*180/math.pi
    direction_errors={'current_observed_recoil':angle(direction(angles)),'learned_motor':angle(direction(angles+forecast[:,3:]*5))}
    def calibration(prob,mask):
        if not bool(mask.any()):return None
        q=prob[mask].clamp(1e-6,1-1e-6);t=y[mask]
        bins=[]
        for i in range(5):
            b=(q>=i*.2)&(q<=(i+1)*.2 if i==4 else q<(i+1)*.2)
            if bool(b.any()):bins.append(dict(low=i*.2,high=(i+1)*.2,shots=int(b.sum()),predicted=float(q[b].mean()),actual=float(t[b].mean())))
        return dict(shots=int(mask.sum()),predicted=float(q.mean()),actual=float(t.mean()),
            log_loss=float(torch.nn.functional.binary_cross_entropy(q,t)),brier=float((q-t).square().mean()),bins=bins)
    def summarize(mask):
        if not bool(mask.any()):return None
        return dict(shots=int(mask.sum()),muzzle_mean_error={k:float(v[mask].mean()) for k,v in muzzle_errors.items()},
            direction_mean_error_degrees={k:float(v[mask].mean()) for k,v in direction_errors.items()},
            calibration={k:calibration(v,mask) for k,v in probabilities.items()})
    summaries={};live_errors={}
    for label in mg['variants']:
        mask=torch.tensor([m['label']==label for m in members],dtype=torch.bool,device='cuda')
        ranked=mask&torch.tensor([m['ranked'] for m in members],dtype=torch.bool,device='cuda')
        clear=mask&~torch.tensor([m['applied_aim_changed'] for m in members],dtype=torch.bool,device='cuda')
        changed=mask&torch.tensor([m['applied_motion_changed'] for m in members],dtype=torch.bool,device='cuda')
        age=mask&torch.tensor([m['observation_age_ms']>0 for m in members],dtype=torch.bool,device='cuda')
        summaries[label]=dict(all=summarize(mask),ranked=summarize(ranked),aim_preserved=summarize(clear),motion_changed=summarize(changed),age_positive=summarize(age))
        if label in logits and bool(ranked.any()):
            logged=tensor([m['logged_after_logit'] or 0. for m in members])
            error=float((logits[label][ranked]-logged[ranked]).abs().max());assert error<1e-6,error
            live_errors[label]=error
    for i,m in enumerate(members):
        m['muzzle_error']={k:float(v[i]) for k,v in muzzle_errors.items()}
        m['direction_error_degrees']={k:float(v[i]) for k,v in direction_errors.items()}
        m['probabilities']={k:float(v[i]) for k,v in probabilities.items()}
    with gzip.open(a.out/'members.jsonl.gz','wt',encoding='utf-8') as f:
        for m in members:f.write(json.dumps(m,separators=(',',':'))+'\n')
    compact_closed_exports(a.out)
    for path,h in pins.items():assert sha(Path(path))==h,path
    report=dict(state='complete',version='combat_motor_native_error_calibration_v1',device='cuda',shots=len(members),
        cases=case_counts,variants=summaries,live_score_max_errors=live_errors,inputs=pins,script_sha256=sha(__file__),members_sha256=sha(a.out/'members.jsonl.gz'),
        training_performed=False,promotion=False,scope='Closed validation actual shots only, no fitting. Both predictors evaluated on the same observed actual-action rows within each regime. Native future pose/aim/outcome are diagnostic labels only. Calibration of chosen shots does not label unexecuted angles or identify the causal reason for changed wins.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(json.dumps({k:v['all'] for k,v in summaries.items()},indent=2))


if __name__=='__main__':main()
