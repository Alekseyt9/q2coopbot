"""Expand input parameters and Adam moments on CUDA without an optimizer step.

This is the tensor transfer part, not a complete persisted model migration.
The caller must validate the source model/seal, rebuild the new model, verify
outputs and restoration, then publish fresh weights/checkpoint/seals.
"""
import copy
from train_combat_bc import torch


def expand_checkpoint_inputs(checkpoint, parameter_names, old_width, new_width):
    assert checkpoint['version'] == 'combat_architecture_checkpoint_v1'
    assert isinstance(old_width,int) and isinstance(new_width,int) and 0 < old_width < new_width
    if checkpoint.get('sequence_retention'):
        raise ValueError('Active sequence retention requires an explicit observation/objective migration')
    if checkpoint.get('retention_weights',[1.,1.]) != [0.,0.]:
        raise ValueError('Active anchor retention requires an explicit observation/objective migration')
    result = copy.deepcopy(checkpoint)
    changes = []
    extra = new_width - old_width
    for role in ('actor','value'):
        names = list(parameter_names[role])
        state = checkpoint[role]
        assert names == list(state), 'Only parameter-only modules with pinned parameter order are supported'
        assert all(v.is_cuda and bool(torch.isfinite(v).all()) for v in state.values())
        candidates = [name for name,v in state.items() if v.ndim == 2 and v.shape[1] == old_width]
        assert len(candidates) == 1, 'Expected one unambiguous input matrix per module'
        name = candidates[0]
        old = state[name]
        result[role][name] = torch.cat((old,old.new_zeros(old.shape[0],extra)),1)
        optimizer = result[role + '_optimizer']
        assert len(optimizer['param_groups']) == 1
        ids = optimizer['param_groups'][0]['params']
        ordered = names + (['log_std'] if role == 'actor' else [])
        assert len(ids) == len(ordered) and len(set(ids)) == len(ids)
        assert set(optimizer['state']) <= set(ids)
        for identifier,param_name in zip(ids,ordered):
            expected_shape = state[param_name].shape if param_name != 'log_std' else checkpoint['log_std'].shape
            for field,tensor in optimizer['state'].get(identifier,{}).items():
                assert tensor.is_cuda and bool(torch.isfinite(tensor).all())
                if field == 'step':
                    assert tensor.numel() == 1
                    continue
                assert field in ('exp_avg','exp_avg_sq','max_exp_avg_sq') and tensor.shape == expected_shape
                if param_name == name:
                    optimizer['state'][identifier][field] = torch.cat((tensor,tensor.new_zeros(tensor.shape[0],extra)),1)
        changes.append(dict(role=role,parameter=name,old_width=old_width,new_width=new_width))
    return result,changes
