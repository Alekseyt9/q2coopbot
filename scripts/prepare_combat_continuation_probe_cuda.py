"""Seal a first TRAIN-only continuation-fork smoke, with unchanged neural parameters."""
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
    for name in ('collection','protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--window-prepared',type=Path,help='Accepted CUDA five-frame preparation, technical overlap smoke only')
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or expected==h,str(path)
        pins[str(path)]=h;return path
    p=read(pin(a.protocol));assert p['version']=='combat_movement_continuation_stability_v1'
    root=a.collection.resolve();dc=read(pin(root/'dataset/complete.json'))
    ds=read(pin(root/'dataset/report.json',dc['report_sha256']));assert ds['gate_passed']
    records=[r for r in read(pin(root/'dataset/records.json',dc['records_sha256']))['records'] if r['split']=='train']
    selected=[]
    for mp in sorted({r['map'] for r in records}):
        selected.append(min([r for r in records if r['map']==mp],key=lambda r:r['seed']))
    assert len(selected)==4
    record=selected[0];context=record['context']
    assert bool(a.window_prepared)==bool(p.get('window_acceptance'))
    if a.window_prepared:
        proof_path=Path(p['window_acceptance']);proof_seal=read(pin(proof_path/'complete.json'))
        proof=read(pin(proof_path/'report.json',proof_seal['report_sha256']))
        assert proof['baseline_full_exact'] and proof['admitted']==5 and proof['excluded']==0
        assert proof['context']==context and proof['source_weights_sha256']==p['source_weights_sha256']
    old=a.window_prepared or root/'preparations'/context['context_id'];old_seal=read(pin(old/'complete.json'))
    old_p=read(pin(old/'report.json',old_seal['report_sha256']))
    for path,h in old_p['inputs'].items():pin(path,h)
    source=Path(old_p['source']);source_seal=read(pin(source/'complete.json'))
    model=read(pin(source/'weights.json',source_seal['weights_sha256']));pin(source/'checkpoint.pt',source_seal['checkpoint_sha256'])
    assert source_seal['weights_sha256']==p['source_weights_sha256']==ds['source_weights_sha256']
    fork=dict(version=VERSION,fork_frame=context['frame']+1,seed=p['continuation_seeds'][0])
    movement={v['label']:v for v in old_p['variants']}
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    original_actor=branch(base_model(model,allow_shared=True),'actor').requires_grad_(False)
    original_joint=JointAim(model,allow_shared=True).requires_grad_(False)
    variants=[];a.out.mkdir(parents=True)
    for index,(label,move,with_fork) in enumerate([('control','control',True),('ordinary_reference',None,True),('left','left_five' if a.window_prepared else 'left',True),('ordinary_no_fork',None,False)]):
        m=read(pin(movement[move]['model'],movement[move]['model_sha256'])) if move else copy.deepcopy(model)
        if with_fork:m['diagnostic_continuation_probe']=fork
        validate(m)
        if with_fork:
            try:base_model(m,allow_shared=True)
            except AssertionError:pass
            else:raise AssertionError('Fork accepted by ordinary model loader')
        actor=branch(base_model(m,allow_probe=True,allow_shared=True),'actor').requires_grad_(False)
        joint=JointAim(m,allow_probe=True,allow_shared=True).requires_grad_(False)
        assert equal_tree(actor.state_dict(),original_actor.state_dict()) and equal_tree(joint.state_dict(),original_joint.state_dict())
        stripped=copy.deepcopy(m);stripped.pop('diagnostic_continuation_probe',None);stripped.pop('diagnostic_movement_probe',None)
        assert stripped==model
        folder=a.out/'models'/label;folder.mkdir(parents=True);save(folder/'weights.json',m)
        variants.append(dict(label=label,index=index,model=str((folder/'weights.json').resolve()),model_sha256=sha(folder/'weights.json'),
            mean_shift=m.get('diagnostic_movement_probe',{}).get('mean_shift'),frame=context['frame'],fork=with_fork))
    for path,h in pins.items():assert sha(path)==h,path
    report=dict(state='complete',version='combat_continuation_probe_preparation_v1',device='cuda',diagnostic_only=True,
        context=context,contexts_selected=[r['context'] for r in selected],variants=variants,source=str(source),
        source_weights_sha256=p['source_weights_sha256'],registry=old_p['registry'],baseline=old_p['baseline'],
        original_collection=str(root),protocol=str(a.protocol.resolve()),protocol_sha256=sha(a.protocol),fork=fork,
        inputs=pins,neural_parameters_exact=True,training_performed=False,validation_used=False,confirmation_used=False,promotion=False)
    if a.window_prepared:report.update(window_smoke=True,movement_duration_frames=5)
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(context=context['context_id'],fork=fork,variants=len(variants)))


if __name__=='__main__':main()
