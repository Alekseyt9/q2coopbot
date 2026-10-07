"""Differentiable native planar input basis for CUDA combat training.

Continuous command approximation; no collision, velocity or trajectory model.
Actions are forward, side, yaw_delta/180, pitch_delta/180.
"""
import math
from ppo_combat import torch


def planar_world_input(features, actions):
    assert features.is_cuda and actions.is_cuda, 'CUDA required'
    yaw = torch.atan2(features[..., 6], features[..., 7]) + actions[..., 2] * math.pi
    pitch = (torch.atan2(features[..., 8], features[..., 9]) + actions[..., 3] * math.pi).clamp(-math.radians(89), math.radians(89))
    forward = actions[..., 0] * (pitch / 3).cos()
    side = actions[..., 1]
    return torch.stack((forward * yaw.cos() + side * yaw.sin(),
                        forward * yaw.sin() - side * yaw.cos()), -1)


def coupling_mse(features, predicted, targets, mask):
    assert mask.any(), 'Positive paired movement/aim labels required'
    return (planar_world_input(features, predicted)[mask] -
            planar_world_input(features, targets)[mask]).square().mean()
