"""Four fixed TRAIN contexts times three future policy streams; no neural changes."""
import argparse
import copy
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from ppo_combat import torch,training_devices
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import JointAim,base_model
from combat_continuation_probe import VERSION,validate
from fork_conditional_weapon_exploration_cuda import equal_tree


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','protocol','acceptance','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path)
        h=pins[key] if key in pins else sha(path)
        assert expected is None or expected==h,str(path);pins[key]=h;return path
    p=read(pin(a.protocol));assert p['version']=='combat_movement_continuation_stability_v1'
    window=p.get('study')=='combat_movement_window_comparison_v1'
    if window:
        wp=Path(p['window_continuation_acceptance']);wc=read(pin(wp/'complete.json'))
        wr=read(pin(wp/'report.json',wc['report_sha256']))
        assert wr['gate_passed'] and wr['nofork_full_exact'] and wr['zero_fork_full_exact']
        assert all(r['movement_window']==dict(duration_frames=5,observed_receipts=5) for r in wr['results'])
        assert p['movement_duration_frames']==5 and p['movement_labels']==['control','left','right','forward','backward']
    ac=read(pin(a.acceptance/'complete.json'));ar=read(pin(a.acceptance/'report.json',ac['report_sha256']))
    assert ar['gate_passed'] and ar['nofork_full_exact'] and ar['zero_fork_full_exact'] and ar['fork_prefix_exact']
    for path,h in ar['inputs'].items():pin(path,h)
    root=a.collection.resolve();ds_closed=read(pin(root/'dataset/complete.json'))
    ds=read(pin(root/'dataset/report.json',ds_closed['report_sha256']));assert ds['gate_passed']
    records=[r for r in read(pin(root/'dataset/records.json',ds_closed['records_sha256']))['records'] if r['split']=='train']
    selected=[min([r for r in records if r['map']==mp],key=lambda r:r['seed']) for mp in sorted({r['map'] for r in records})]
    assert len(selected)==4 and len(p['continuation_seeds'])==p['continuation_replicates']
    assert len(set(p['continuation_seeds']))==len(p['continuation_seeds']) and all(type(seed) is int and 0<=seed<=2**31-1 for seed in p['continuation_seeds'])
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    a.out.mkdir(parents=True);units=[];count=0
    for record in selected:
        context=record['context'];old=root/'preparations'/context['context_id']
        os_=read(pin(old/'complete.json'));old_p=read(pin(old/'report.json',os_['report_sha256']))
        for path,h in old_p['inputs'].items():pin(path,h)
        source=Path(old_p['source']);ss=read(pin(source/'complete.json'));model=read(pin(source/'weights.json',ss['weights_sha256']))
        assert ss['weights_sha256']==p['source_weights_sha256']
        actor=branch(base_model(model,allow_shared=True),'actor').requires_grad_(False)
        joint=JointAim(model,allow_shared=True).requires_grad_(False)
        old_models={v['label']:read(pin(v['model'],v['model_sha256'])) for v in old_p['variants']}
        order=[v['label'] for v in old_p['variants']]
        if window:order=list(p['movement_labels']);assert set(order)<=old_models.keys()
        for replicate,seed in enumerate(p['continuation_seeds']):
            unit_id=f"{context['context_id']}-fork-{seed}";target=a.out/'units'/unit_id;target.mkdir(parents=True)
            fork=dict(version=VERSION,fork_frame=context['frame']+1,seed=seed)
            names=['control','ordinary_reference']+[name for name in order if name!='control']
            if replicate==0:names.append('ordinary_no_fork')
            variants=[]
            for label in names:
                changed=copy.deepcopy(old_models[label] if label in old_models else model)
                if window and changed.get('diagnostic_movement_probe'):
                    from combat_movement_probe import WINDOW_VERSION,validate as validate_movement
                    changed['diagnostic_movement_probe'].update(version=WINDOW_VERSION,duration_frames=5)
                    validate_movement(changed)
                with_fork=label!='ordinary_no_fork'
                if with_fork:changed['diagnostic_continuation_probe']=fork
                validate(changed)
                decoded=branch(base_model(changed,allow_probe=True,allow_shared=True),'actor').requires_grad_(False)
                decoded_joint=JointAim(changed,allow_probe=True,allow_shared=True).requires_grad_(False)
                assert equal_tree(decoded.state_dict(),actor.state_dict()) and equal_tree(decoded_joint.state_dict(),joint.state_dict())
                core=copy.deepcopy(changed);core.pop('diagnostic_movement_probe',None);core.pop('diagnostic_continuation_probe',None);assert core==model
                folder=target/'models'/label;folder.mkdir(parents=True);save(folder/'weights.json',changed)
                variants.append(dict(label=label,index=len(variants),model=str((folder/'weights.json').resolve()),model_sha256=sha(folder/'weights.json'),
                    mean_shift=changed.get('diagnostic_movement_probe',{}).get('mean_shift'),frame=context['frame'],fork=with_fork))
            report=dict(state='complete',version='combat_continuation_probe_preparation_v1',device='cuda',diagnostic_only=True,
                unit_id=unit_id,context=context,variants=variants,source=str(source),source_weights_sha256=p['source_weights_sha256'],
                registry=old_p['registry'],baseline=old_p['baseline'],original_collection=str(root),protocol=str(a.protocol.resolve()),
                protocol_sha256=sha(a.protocol),fork=fork,inputs=dict(pins),neural_parameters_exact=True,
                training_performed=False,validation_used=False,confirmation_used=False,promotion=False)
            if window:report.update(window_smoke=False,movement_duration_frames=5)
            save(target/'report.json',report);save(target/'complete.json',dict(state='complete',report_sha256=sha(target/'report.json')))
            units.append(dict(unit_id=unit_id,prepared=str(target.resolve()),context=context,fork=fork,replicate=replicate+1,
                report_sha256=sha(target/'report.json'),variants=len(variants)))
            count+=len(variants);save(a.out/'progress.json',dict(stage='cuda-preparation',units=len(units),cases=count))
    for path,h in pins.items():assert sha(path)==h,path
    assert len(units)==4*len(p['continuation_seeds']) and count<=p.get('native_cases',96)
    if p.get('native_cases'):assert count==p['native_cases']==p['cuda_cases']
    report=dict(state='complete',version='combat_continuation_units_preparation_v1',device='cuda',units=units,cases=count,
        protocol_sha256=sha(a.protocol),source_weights_sha256=p['source_weights_sha256'],inputs=pins,training_performed=False,promotion=False)
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    save(a.out/'progress.json',dict(stage='complete',units=len(units),cases=count));print(dict(units=len(units),cases=count))


if __name__=='__main__':main()
