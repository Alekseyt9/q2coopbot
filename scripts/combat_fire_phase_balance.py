"""Observed weapon-cycle strata for an explicit fire-only PPO loss fork.

This changes the TRAIN objective, never inference or recorded rewards/actions.
The two observed state strata receive equal mass; no action labels are added.
"""
from ppo_combat import torch
from combat_body_features import legacy

VERSION='observed_hand_grenade_hold11_equal_mass_v1'


def weights(features, version):
    assert version==VERSION, 'Unknown fire phase weighting'
    assert features.is_cuda and features.ndim==2
    x=legacy(features)
    # Common20 column 5 is observed gun_frame / 30. Equipped-weapon tail
    # starts at 1121: category 11 is the observed hand grenade, not a request.
    held=(x[:,1121+11]>.5)&((x[:,5]-11/30).abs()<1e-6)
    count=int(held.sum());n=len(x)
    assert 0<count<n, 'Balanced fire fitting requires both observed state strata'
    result=torch.where(held, n/(2*count), n/(2*(n-count))).to(features.dtype)
    assert bool(torch.isfinite(result).all()) and bool((result>0).all())
    return result,dict(version=VERSION,rows=n,grenade_hold11_rows=count,other_rows=n-count,
        grenade_hold11_weight=n/(2*count),other_weight=n/(2*(n-count)),
        basis='Observed equipped weapon and gun frame only; no outcome or chosen-action gate')
