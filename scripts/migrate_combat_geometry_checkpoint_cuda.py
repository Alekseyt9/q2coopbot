"""Build an auditable geometry PPO fork; never reinterpret an eval wrapper as resume."""
import argparse
import copy
import json
from pathlib import Path
import shutil
from evaluate_registered_combat_models import read,save,sha
from ppo_combat import torch,nn,training_devices
from ppo_recurrent import durable_write
from combat_joint_aim_policy import JointAim,base_model
from combat_geometry_ppo import GeometryJoint,TRAINING_VERSION
from combat_weapon_state_residual import optimizer as actor_optimizer
from migrate_combat_navigation_cuda import branch
from fork_conditional_weapon_exploration_cuda import equal_tree
from train_combat_joint_aim_cuda import cuda_tree


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('source','runtime','evaluation','dataset','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    ss=read(pin(a.source/'complete.json'))
    for n,k in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):pin(a.source/n,ss[k])
    rs=read(pin(a.runtime/'complete.json'))
    for n,k in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):pin(a.runtime/n,rs[k])
    model=read(a.runtime/'weights.json');source_model=read(a.source/'weights.json')
    assert base_model(model,allow_shared=True)==base_model(source_model)
    restored=copy.deepcopy(model);restored['aim_candidate_head'].pop('shared_scorer');restored['aim_candidate_head']['sampler']=source_model['aim_candidate_head']['sampler']
    assert restored==source_model,'Changes outside new geometry scorer'
    es=read(pin(a.evaluation/'complete.json'));fixture=read(pin(a.evaluation/'fixture-proof.json'))
    binding=next(b for b in fixture['models'] if b['model_sha256']==rs['weights_sha256'])
    assert binding['model']==str((a.runtime/'weights.json').resolve())
    label=binding['label'];proofs=[]
    for p in sorted((a.evaluation/('processing-'+label)).glob('case-*/cuda/cuda-verification.json')):
        r=read(pin(p));assert r['state']=='passed' and r['device']=='cuda' and r['model_sha256']==rs['weights_sha256']
        assert r['shared_candidate_precision']=='float64' and r['frame_hash_candidate_assignments_verified'];proofs.append(r)
    assert len(proofs)==8 and es['cuda_cases']==24
    ds=read(pin(a.dataset/'complete.json'));meta=read(pin(a.dataset/'report.json',ds['report_sha256']))
    rows=[r for r in map(json.loads,pin(a.dataset/'contexts.jsonl',ds['labels_sha256']).open(encoding='utf-8-sig')) if r['split']=='train'][:8]
    assert len(rows)==8
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cp=torch.load(a.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    assert cp['weights_sha256']==ss['weights_sha256']
    core=base_model(model,allow_shared=True)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        actor,value,joint=branch(core,'actor'),branch(core,'value'),GeometryJoint(model)
        reference=JointAim(model,allow_shared=True).requires_grad_(False)
    assert equal_tree(actor.state_dict(),cp['actor']) and equal_tree(value.state_dict(),cp['value'])
    assert equal_tree(joint.head.state_dict(),cp['aim_candidate_head'])
    std=nn.Parameter(torch.tensor(model['log_std'],device='cuda'));assert torch.equal(std,cp['log_std'])
    rates=[g['lr'] for g in cp['actor_optimizer']['param_groups']];assert rates==[3e-5,.0003,.0003,.0003]
    ao=actor_optimizer(actor,std,*rates);ao.load_state_dict(copy.deepcopy(cp['actor_optimizer']))
    vo=torch.optim.Adam(value.parameters(),lr=cp['config']['value_lr']);vo.load_state_dict(copy.deepcopy(cp['value_optimizer']))
    assert equal_tree(cuda_tree(ao.state_dict()),cp['actor_optimizer']) and equal_tree(cuda_tree(vo.state_dict()),cp['value_optimizer'])
    opt=joint.geometry_optimizer();assert not opt.state_dict()['state']
    assert sum(p.numel() for p in joint.geometry_parameters())==266
    x=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float32,requires_grad=True)
    z=torch.tensor([r['sample']['aim_candidate']['source_latent'] for r in rows],device='cuda',dtype=torch.float32,requires_grad=True)
    cat=lambda k:torch.tensor([r['sample'].get(k,0) for r in rows],device='cuda',dtype=torch.long)
    timing=torch.tensor([r['sample']['aim_candidate']['timing'] for r in rows],device='cuda',dtype=torch.float32)
    args=(x,z,cat('target'),cat('aim_mode'),cat('vertical'),cat('weapon'),timing)
    with torch.no_grad():before,active=joint(*args);expected,ea=reference(*args)
    assert torch.equal(before,expected) and torch.equal(active,ea) and bool(active.all())
    initial=joint.geometry_state();legacy_before=copy.deepcopy(joint.head.state_dict());hints_before=copy.deepcopy(joint.hints.state_dict())
    probe=copy.deepcopy(joint);probe_opt=probe.geometry_optimizer()
    logits,_=probe(*args);loss=-logits.log_softmax(-1)[:,0].mean();loss.backward()
    assert x.grad is None and z.grad is None,'Pathwise gradient through recorded action/context'
    assert probe.shared_beta.grad is not None and bool(torch.isfinite(probe.shared_beta.grad)) and float(probe.shared_beta.grad.abs())>0
    assert any(p.grad is not None and float(p.grad.abs().sum())>0 for p in probe.shared.parameters())
    assert all(p.grad is None for p in probe.head.parameters()) and all(p.grad is None for p in probe.hints.parameters())
    grad=float(nn.utils.clip_grad_norm_(probe.geometry_parameters(),.5));assert grad>0
    probe_opt.step();probe.project()
    assert not equal_tree(initial,probe.geometry_state()) and equal_tree(legacy_before,probe.head.state_dict()) and equal_tree(hints_before,probe.hints.state_dict())
    assert equal_tree(initial,joint.geometry_state()) and not opt.state_dict()['state']
    a.out.mkdir(parents=True);durable_write(a.out/'weights.json',lambda f:f.write((a.runtime/'weights.json').read_bytes()))
    # Deserialization on CPU below preserves serialized CPU RNG bytes. Neural
    # evaluation, gradient checks and comparisons above/below all use CUDA.
    fork=torch.load(a.source/'checkpoint.pt',map_location='cpu',weights_only=True)
    fork.update(weights_sha256=sha(a.out/'weights.json'),geometry_aim_head=joint.geometry_state(),geometry_aim_optimizer=opt.state_dict(),
        geometry_aim_training=dict(version=TRAINING_VERSION,updates_completed=0,source_weights_sha256=ss['weights_sha256'],
            initializer_weights_sha256=rs['weights_sha256'],head_lr=.0003,actor_group_lrs=rates,legacy_head_inactive=True,
            actor_critic_optimizer_rng_preserved=True,head_optimizer_reinitialized_for_ppo=True))
    durable_write(a.out/'checkpoint.pt',lambda f:torch.save(fork,f))
    loaded=torch.load(a.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    for k in cp:
        if k!='weights_sha256':assert equal_tree(cp[k],loaded[k]),k
    assert equal_tree(initial,loaded['geometry_aim_head']) and not loaded['geometry_aim_optimizer']['state']
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):decoded=GeometryJoint(read(a.out/'weights.json'))
    decoded.load_geometry_state(loaded['geometry_aim_head']);decoded_opt=decoded.geometry_optimizer();decoded_opt.load_state_dict(loaded['geometry_aim_optimizer'])
    with torch.no_grad():after,other=decoded(*args)
    assert torch.equal(before,after) and torch.equal(active,other)
    assert equal_tree(cuda_tree(decoded_opt.state_dict()),loaded['geometry_aim_optimizer'])
    assert decoded.export(model)==model
    for p,h in pins.items():assert sha(p)==h,p
    snapshot=a.out/'sources';snapshot.mkdir()
    for n in (Path(__file__).name,'combat_geometry_ppo.py','combat_joint_aim_policy.py'):shutil.copy2(Path(__file__).with_name(n),snapshot/n)
    report=dict(state='complete',version='combat_geometry_checkpoint_migration_v1',device='cuda',head_parameters=266,source=str(a.source.resolve()),
         source_weights_sha256=ss['weights_sha256'],weights_sha256=sha(a.out/'weights.json'),actor_value_std_adam_rng_history_exact=True,
         new_head_optimizer_empty=True,head_gradient_verified=True,recorded_action_pathwise_gradient_absent=True,checkpoint_roundtrip_exact=True,
         evaluation_policy_unchanged=True,gradient_test_rows=len(rows),gradient_test_head_norm=grad,gradient_test_step_discarded=True,
         training_performed=False,ppo_update_performed=False,ppo_trainer_integrated=False,promotion=False,inputs=pins,script_sha256=sha(__file__),
         scope='Development initialization only. Real source body/critic/optimizers/RNG plus new geometry head; no evaluation wrapper treated as resume, no evaluation labels trained.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',weights_sha256=sha(a.out/'weights.json'),checkpoint_sha256=sha(a.out/'checkpoint.pt'),report_sha256=sha(a.out/'report.json')))
    print(json.dumps({k:v for k,v in report.items() if k!='inputs'},indent=2))


if __name__=='__main__':main()
