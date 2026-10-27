"""Equal battle mass for CUDA full-policy PPO; no outcome-based weights."""
import copy
from collections import Counter
from ppo_combat import torch
from combat_full_joint_ppo import SCOPE

VERSION='combat_equal_episode_mass_v1'


def weights(rows):
    assert torch.cuda.is_available() and rows
    counts=Counter(r['seed'] for r in rows)
    n=len(rows);k=len(counts)
    w=torch.tensor([n/(k*counts[r['seed']]) for r in rows],device='cuda',dtype=torch.float32)
    assert bool(torch.isfinite(w).all()) and bool((w>0).all())
    assert abs(float(w.mean())-1)<1e-5
    return w,dict(version=VERSION,cases=k,rows=n,rows_per_case={str(s):v for s,v in sorted(counts.items())},
        minimum_weight=float(w.min()),maximum_weight=float(w.max()),outcome_independent=True)


def normalize_advantages(advantages,w):
    assert advantages.is_cuda and w.is_cuda and advantages.shape==w.shape
    mass=w.sum();mean=(advantages*w).sum()/mass
    variance=((advantages-mean).square()*w).sum()/mass
    return (advantages-mean)/(variance.sqrt()+1e-8)


def fork_config(scope,config):
    assert scope.scope==SCOPE and scope.config.get('actor_scope','all')=='all'
    assert config.get('actor_scope','all')=='all' and config['version']==SCOPE
    assert config.get('episode_balance')==VERSION and scope.config.get('episode_balance') is None
    assert config.get('fire_phase_balance') is None
    changed={k for k in scope.config.keys()|config.keys() if scope.config.get(k)!=config.get(k)}
    assert changed=={'episode_balance'},'Episode fork only changes battle weighting'
    scope.config=copy.deepcopy(config)
    scope.verify()
