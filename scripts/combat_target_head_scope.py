"""Projected target-row updates preserving other actor parameters and Adam state."""
import copy
from ppo_combat import torch, nn


def freeze_target_scope(actor, std, optimizer):
    assert next(actor.parameters()).is_cuda and std.is_cuda
    optimizer.zero_grad(set_to_none=True)
    core = getattr(actor, 'base', actor)
    modules = [core[-1]] if isinstance(core, nn.Sequential) else [core.head, core.output if hasattr(core, 'gru') else core.residual]
    selected = {}
    for module in modules:
        assert isinstance(module, nn.Linear) and module.out_features in (45, 81)
        for parameter in (module.weight, module.bias):
            mask = torch.zeros_like(parameter, dtype=torch.bool)
            mask[20:29] = True
            selected[parameter] = mask
    for parameter in list(actor.parameters()) + [std]:
        parameter.requires_grad_(parameter in selected)
        if parameter in selected:
            parameter.register_hook(lambda gradient, mask=selected[parameter]: gradient.masked_fill(~mask, 0))
    parameters = list(actor.parameters()) + [std]
    frozen = {p: p.detach().clone() for p in parameters}
    states = {p: copy.deepcopy(optimizer.state.get(p, {})) for p in parameters}

    def project():
        # Adam momentum can move zero-gradient rows. Restore physical outputs
        # and their moments after every proposal, including repaired fallbacks.
        with torch.no_grad():
            for p in parameters:
                if p not in selected:
                    p.copy_(frozen[p])
                    if states[p]:
                        optimizer.state[p] = copy.deepcopy(states[p])
                    else:
                        optimizer.state.pop(p, None)
                    continue
                mask = selected[p]
                p[~mask] = frozen[p][~mask]
                for name, value in optimizer.state.get(p, {}).items():
                    if torch.is_tensor(value) and value.shape == p.shape:
                        initial = states[p].get(name)
                        value[~mask] = initial[~mask] if initial is not None else 0
    return project, sum(int(mask.sum()) for mask in selected.values())
