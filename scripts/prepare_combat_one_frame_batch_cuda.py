"""Seal multiple scheduled native pair contexts in one CUDA acquisition batch."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from ppo_combat import torch,training_devices
from combat_joint_aim_policy import JointAim,validate,PROBE_VERSION,POLICY_PROBE_VERSION,events
from combat_aim_candidates import choose,frame_uniforms,OFFSETS
from fork_conditional_weapon_exploration_cuda import equal_tree


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('selection','out'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--context-id',help='Prepare one explicitly selected smoke context')
    ap.add_argument('--exclude-context',action='append',default=[],help='Exclude an already acquired context; no hidden repeated games')
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    selected_seal=read(pin(a.selection/'complete.json'));selection=read(pin(a.selection/'report.json',selected_seal['report_sha256']))
    assert selection['version'] in ('combat_late_aim_selection_v1','combat_late_aim_selection_v2_policy') and not selection['training_performed']
    policy_probe=selection['version']=='combat_late_aim_selection_v2_policy'
    probe_version=POLICY_PROBE_VERSION if policy_probe else PROBE_VERSION
    for p,h in selection['inputs'].items():pin(p,h)
    baseline=Path(selection['baseline']);source=Path(selection['source']);seal=read(pin(source/'complete.json'))
    model=read(pin(source/'weights.json',seal['weights_sha256']));pin(source/'checkpoint.pt',seal['checkpoint_sha256'])
    assert seal['weights_sha256']==selection['source_weights_sha256'] and not model.get('diagnostic_joint_aim_probe')
    contexts=copy.deepcopy(selection['contexts']);assert 1<=len(contexts)<=32
    identifiers={c['context_id'] for c in contexts}
    assert set(a.exclude_context)<=identifiers and (not a.context_id or a.context_id in identifiers)
    assert not (a.context_id and a.exclude_context)
    contexts=[c for c in contexts if (not a.context_id or c['context_id']==a.context_id) and c['context_id'] not in a.exclude_context]
    assert contexts
    heldout=set(selection['heldout_seeds']);assert all((c['seed'] in heldout)==(c['split']=='development') for c in contexts)
    cases={}
    for c in contexts:
        if c['job'] in cases:continue
        folder=baseline/'processing'/f"case-{c['job']:04d}"/'cuda';meta=read(pin(folder/'report.json'))
        rows=list(map(json.loads,pin(folder/'rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig')))
        assert all(r['seed']==c['seed'] for r in rows);cases[c['job']]=dict(rows=rows,byframe={r['frame']:r for r in rows})
    selected_rows=[cases[c['job']]['byframe'][c['frame']] for c in contexts]
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):ordinary=JointAim(model,allow_shared=True).requires_grad_(False)
    frozen=copy.deepcopy(ordinary.state_dict())
    x=torch.tensor([r['features'] for r in selected_rows],device='cuda',dtype=torch.float32)
    cat=lambda n:torch.tensor([r['sample'].get(n,0) for r in selected_rows],device='cuda',dtype=torch.long)
    z,old,timing,active=events(selected_rows,x,cat('target'),cat('aim_mode'),cat('vertical'),cat('weapon'))
    assert bool(active.all()) and bool((old==0).all())
    captured=[];handle=ordinary.hints.register_forward_hook(lambda m,args,out:captured.append(out.detach()))
    with torch.inference_mode():ordinary(x,z,cat('target'),cat('aim_mode'),cat('vertical'),cat('weapon'),timing)
    handle.remove();hints=captured[0].reshape(len(contexts),25).float().clone()
    a.out.mkdir(parents=True);variants=[]
    save(a.out/'progress.json',dict(stage='cuda-model-preparation',contexts=len(contexts)))
    for ci,c in enumerate(contexts):
        top=[k for k in torch.argsort(hints[ci],descending=True,stable=True).tolist() if k][:2]
        other=sorted((k for k in range(1,25) if k not in top),key=lambda k:hashlib.sha256(f"{c['seed']}:{c['frame']}:{k}".encode()).hexdigest())[:2]
        c.update(categories=[0,*top,*other],predicted_hit_probabilities=hints[ci].double().sigmoid().tolist(),
            purpose='development_labels_only' if c['split']=='development' else 'train_acquisition')
        rows=cases[c['job']]['rows'];xx=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float32)
        cc=lambda n:torch.tensor([r['sample'].get(n,0) for r in rows],device='cuda',dtype=torch.long)
        zz,old_choices,tt,aa=events(rows,xx,cc('target'),cc('aim_mode'),cc('vertical'),cc('weapon'))
        frames=torch.tensor([r['frame'] for r in rows],device='cuda',dtype=torch.long);at=frames==c['frame'];assert int(at.sum())==1
        uniforms=torch.tensor([frame_uniforms(r['sample']['sampling_seed'],r['frame'],1) for r in rows],device='cuda',dtype=torch.float64)
        with torch.no_grad():reference,reference_active=ordinary(xx,zz,cc('target'),cc('aim_mode'),cc('vertical'),cc('weapon'),tt)
        assert torch.equal(reference_active,aa) and torch.equal(choose(reference.double(),uniforms[:,0],uniforms[:,1],aa),old_choices)
        module=None
        for k in c['categories']:
            logits=model['aim_candidate_head']['layers'][-1]['bias']
            if policy_probe:logits=reference[at][0].tolist()
            if k:
                q=torch.full((25,),1e-8,device='cuda',dtype=torch.float64);q[k]=1-24e-8;logits=q.log().float().tolist()
            updated=copy.deepcopy(model);updated['diagnostic_joint_aim_probe']=dict(version=probe_version,frame=c['frame'],logits=logits)
            validate(updated,allow_probe=True,allow_shared=True)
            if module is None:
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):module=JointAim(updated,allow_probe=True,allow_shared=True).requires_grad_(False)
            with torch.no_grad():
                module.probe_logits.copy_(torch.tensor(logits,device='cuda',dtype=torch.float32))
                scored,actual_active=module(xx,zz,cc('target'),cc('aim_mode'),cc('vertical'),cc('weapon'),tt,frames=frames)
                assert torch.equal(actual_active,aa)
                assert torch.equal(scored[~at],reference[~at])
                picked=choose(scored.double(),uniforms[:,0],uniforms[:,1],aa);expected=old_choices.clone();expected[at]=k
                assert torch.equal(picked,expected),(c['context_id'],k)
            stripped=copy.deepcopy(updated);stripped.pop('diagnostic_joint_aim_probe');assert stripped==model
            directory=a.out/'models'/f'context-{ci:02d}-category-{k:02d}';directory.mkdir(parents=True);save(directory/'weights.json',updated)
            variants.append(dict(context_id=c['context_id'],category=k,model=str((directory/'weights.json').resolve()),model_sha256=sha(directory/'weights.json'),
                episode=c['episode'],seed=c['seed'],frame=c['frame'],reference_job=c['job'],offset=c['offset'],instance=c['instance'],
                dataset_split=c['split'],purpose=c['purpose'],offset_yaw_degrees=OFFSETS[k//5],offset_pitch_degrees=OFFSETS[k%5]))
        assert equal_tree({k:v for k,v in module.state_dict().items() if k!='probe_logits'},ordinary.state_dict())
    assert equal_tree(frozen,ordinary.state_dict())
    for p,h in pins.items():assert sha(Path(p))==h,p
    snapshot=a.out/'sources';snapshot.mkdir();__import__('shutil').copy2(__file__,snapshot/Path(__file__).name)
    report=dict(state='complete',version='combat_one_frame_aim_batch_preparation_v2_policy' if policy_probe else 'combat_one_frame_aim_batch_preparation_v1',probe_version=probe_version,device='cuda',diagnostic_only=True,
        baseline=str(baseline.resolve()),source=str(source.resolve()),source_weights_sha256=seal['weights_sha256'],registry=selection['registry'],
        protocol=selection['protocol'],protocol_sha256=selection['protocol_sha256'],contexts=contexts,variants=variants,
        selection=str(a.selection.resolve()),selection_report_sha256=selected_seal['report_sha256'],heldout_seeds=selection['heldout_seeds'],
        explicit_context_filter=a.context_id,excluded_contexts=a.exclude_context,
        checkpoint_policy='Shared immutable baseline checkpoint; acquisition copies are metadata-only models, not training checkpoints',
        inputs=pins,script_sha256=sha(__file__),training_performed=False,validation_used=False,promotion=False,
        scope='All selected train/development acquisition contexts batched. Development labels stay development; no policy training or quality claim.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    save(a.out/'progress.json',dict(stage='complete',contexts=len(contexts),native_cases_planned=len(variants)))
    print(json.dumps(dict(contexts=len(contexts),variants=len(variants),train=sum(c['split']=='train' for c in contexts),development=sum(c['split']=='development' for c in contexts))))


if __name__=='__main__':main()
