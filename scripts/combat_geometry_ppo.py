"""Explicit training adapter for the geometry-delta categorical head."""
import copy
from ppo_combat import torch,nn,layers
from combat_joint_aim_policy import JointAim,GEOMETRY_KIND,GEOMETRY_SAMPLER

TRAINING_VERSION='combat_geometry_delta_ppo_v1'


class GeometryJoint(JointAim):
    def __init__(self,model):
        assert not model.get('diagnostic_joint_aim_probe') and model.get('diagnostic_movement_probe') is None and model.get('diagnostic_continuation_probe') is None and model.get('diagnostic_weapon_probe') is None
        assert model['aim_candidate_head']['shared_scorer']['kind']==GEOMETRY_KIND
        assert model['aim_candidate_head']['sampler']==GEOMETRY_SAMPLER
        super().__init__(model,allow_shared=True)
        self.head.requires_grad_(False)
        self.hints.requires_grad_(False)
        beta=self.shared_beta.detach().clone()
        del self._buffers['shared_beta']
        self.register_parameter('shared_beta',nn.Parameter(beta))

    def geometry_parameters(self):return [*self.shared.parameters(),self.shared_beta]

    def geometry_state(self):return dict(network=copy.deepcopy(self.shared.state_dict()),beta=self.shared_beta.detach().clone())

    def load_geometry_state(self,state):
        self.shared.load_state_dict(state['network'])
        with torch.no_grad():self.shared_beta.copy_(state['beta'])

    def geometry_optimizer(self,lr=.0003):
        return torch.optim.Adam(self.geometry_parameters(),lr=lr)

    def project(self):
        with torch.no_grad():self.shared_beta.clamp_(-20,20)

    def export(self,model):
        result=copy.deepcopy(model)
        spec=result['aim_candidate_head']['shared_scorer']
        spec.update(layers=layers(self.shared),beta=float(self.shared_beta.detach()))
        return result
