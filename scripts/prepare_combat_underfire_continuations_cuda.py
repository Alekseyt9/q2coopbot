"""Freeze shared-action under-fire windows with three independent future streams."""
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
    for name in ('protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or expected==h,str(path);pins[key]=h;return path
    p=read(pin(a.protocol));assert p['version']=='combat_movement_continuation_stability_v1' and p['study'] in ('combat_underfire_horizon_pilot_v1','combat_underfire_constrained_confirmation_v1','combat_underfire_full_task_replication_v1')
    full_task=p['study']=='combat_underfire_full_task_replication_v1'
    constrained=p['study'] in ('combat_underfire_constrained_confirmation_v1','combat_underfire_full_task_replication_v1');selection={}
    if constrained:
        folder=Path(p['selection_out']);seal=read(pin(folder/'complete.json'))
        selected=read(pin(folder/'report.json',seal['report_sha256']))
        assert selected['version']==('combat_underfire_full_task_selection_v1' if full_task else 'combat_underfire_constrained_selection_v1') and selected['protocol_sha256']==sha(a.protocol)
        for path,h in selected['inputs'].items():pin(path,h)
        selection={item['context']['context_id']:item for item in selected['units']}
    root=Path(p['prepared_index']);closed=read(pin(root/'complete.json'))
    index=read(pin(root/'report.json',closed['report_sha256']))
    assert index['version']=='combat_movement_units_preparation_v1' and index['device']=='cuda' and len(index['units'])==4
    for path,h in index['inputs'].items():pin(path,h)
    assert len(p['continuation_seeds'])==len(set(p['continuation_seeds']))==p['continuation_replicates']==(2 if constrained and not full_task else 3)
    assert all(type(seed) is int and 0<=seed<=2**31-1 for seed in p['continuation_seeds'])
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    a.out.mkdir(parents=True);units=[];count=0
    for old_unit in index['units']:
        old=Path(old_unit['prepared']);seal=read(pin(old/'complete.json'))
        prep=read(pin(old/'report.json',seal['report_sha256']));context=prep['context']
        assert context==old_unit['context'] and context['split']=='train' and context['role']=='auxiliary_train'
        assert prep['neural_parameters_exact'] and prep['zero_control_exact'] and prep['ordinary_outside_window_exact']
        for path,h in prep['inputs'].items():pin(path,h)
        source=Path(prep['source']);model=read(pin(source/'weights.json',p['source_weights_sha256']))
        actor=branch(base_model(model,allow_shared=True),'actor').requires_grad_(False)
        joint=JointAim(model,allow_shared=True).requires_grad_(False)
        old_models={v['label']:read(pin(v['model'],v['model_sha256'])) for v in prep['variants']}
        assert list(old_models)==p['movement_labels']
        for replicate,seed in enumerate(p['continuation_seeds']):
            uid=f"{context['context_id']}-fork-{seed}";target=a.out/'units'/uid;target.mkdir(parents=True)
            fork=dict(version=VERSION,fork_frame=context['frame']+1,seed=seed)
            if constrained:
                item=selection[context['context_id']];assert item['context']==context
                names=['control','ordinary_reference']+([] if item['selected']=='control' else [item['selected']])
            else:names=['control','ordinary_reference']+p['movement_labels'][1:]
            if replicate==0:names.append('ordinary_no_fork')
            variants=[]
            for label in names:
                changed=copy.deepcopy(old_models.get(label,model));with_fork=label!='ordinary_no_fork'
                if with_fork:changed['diagnostic_continuation_probe']=fork
                validate(changed)
                movement=changed.get('diagnostic_movement_probe')
                if movement:assert movement['duration_frames']==p['movement_duration_frames']==5
                decoded=branch(base_model(changed,allow_probe=True,allow_shared=True),'actor').requires_grad_(False)
                decoded_joint=JointAim(changed,allow_probe=True,allow_shared=True).requires_grad_(False)
                assert equal_tree(decoded.state_dict(),actor.state_dict()) and equal_tree(decoded_joint.state_dict(),joint.state_dict())
                core=copy.deepcopy(changed);core.pop('diagnostic_movement_probe',None);core.pop('diagnostic_continuation_probe',None);assert core==model
                directory=target/'models'/label;directory.mkdir(parents=True);save(directory/'weights.json',changed)
                variants.append(dict(label=label,index=len(variants),model=str((directory/'weights.json').resolve()),
                    model_sha256=sha(directory/'weights.json'),mean_shift=movement['mean_shift'] if movement else None,
                    frame=context['frame'],fork=with_fork))
            report=dict(state='complete',version='combat_continuation_probe_preparation_v1',device='cuda',diagnostic_only=True,
                unit_id=uid,context=context,variants=variants,source=str(source),source_weights_sha256=p['source_weights_sha256'],
                registry=prep['registry'],baseline=prep['baseline'],protocol=str(a.protocol.resolve()),protocol_sha256=sha(a.protocol),
                fork=fork,inputs=dict(pins),neural_parameters_exact=True,movement_duration_frames=5,
                training_performed=False,validation_used=False,confirmation_used=False,promotion=False)
            save(target/'report.json',report);save(target/'complete.json',dict(state='complete',report_sha256=sha(target/'report.json')))
            units.append(dict(unit_id=uid,prepared=str(target.resolve()),context=context,fork=fork))
            count+=len(variants);save(a.out/'progress.json',dict(stage='cuda-preparing',units=len(units),cases=count))
    for path,h in pins.items():assert sha(path)==h,path
    assert len(units)==4*p['continuation_replicates'] and count==p['native_cases']==p['cuda_cases']
    report=dict(state='complete',version='combat_continuation_units_preparation_v1',device='cuda',units=units,cases=count,
        protocol_sha256=sha(a.protocol),source_weights_sha256=p['source_weights_sha256'],inputs=pins,training_performed=False,promotion=False)
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    save(a.out/'progress.json',dict(stage='complete',units=len(units),cases=count));print(dict(units=len(units),cases=count))


if __name__=='__main__':main()
