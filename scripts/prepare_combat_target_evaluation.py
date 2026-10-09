"""Small paired native validation for all target-head bootstrap branches."""
import argparse,pathlib,subprocess
from process_combat_architecture_pool import read,sha,save
from prepare_combat_target_heads import build
from ppo_recurrent import torch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',type=pathlib.Path,required=True);ap.add_argument('--controls',type=pathlib.Path,required=True);ap.add_argument('--compiler',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
    assert torch.cuda.is_available() and not a.out.exists()
    audit=read(a.models/'cuda-audit.json');training=read(a.models/'target-bc-report.json');assert audit['state']=='passed' and training['state']=='complete' and len(training['models'])==8
    control=read(a.controls/'protocol.json');fire=next(e for e in control['evaluations'] if e['model']=='firebc');baseline=read(fire['plan'])
    families=[t['episode']['id'] for t in baseline['tasks'][:4]];assert len(families)==4
    a.out.mkdir();entries=[];plans=[];checks=[]
    for report in training['models']:
        folder=a.models/report['model'];weights=folder/'target-bc/weights.json';parent=folder/'weights.json'
        assert sha(weights)==report['weights_sha256'] and sha(parent)==report['parent_sha256']
        assert sha(folder/'target-bc/checkpoint.pt')==report['checkpoint_sha256'] and report['device']=='cuda' and report['shared_parameters_preserved']
        model=read(weights);old=read(parent);cp=torch.load(folder/'target-bc/checkpoint.pt',map_location='cuda',weights_only=True)
        assert cp['weights_sha256']==sha(weights) and cp['parent_sha256']==sha(parent) and cp['epochs']==report['epochs']
        actor=build(model,'actor');reference=build(old,'actor');prefix='head.' if hasattr(actor,'head') else '4.'
        for key,tensor in actor.state_dict().items():
            assert torch.equal(tensor,cp['actor'][key]) and torch.isfinite(tensor).all()
            preserved=tensor[:20] if key.startswith((prefix,'output.','residual.')) else tensor
            prior=reference.state_dict()[key];prior=prior[:20] if key.startswith((prefix,'output.','residual.')) else prior
            assert torch.equal(preserved,prior)
        assert model['value']==old['value'] and model['log_std']==old['log_std']
        if model.get('memory'):assert model['memory']['value']==old['memory']['value']
        if model.get('attention'):assert model['attention']['value']==old['attention']['value']
        checks.append(dict(model=report['model'],checkpoint_equal=True,shared_rows_preserved=True,critic_std_preserved=True))
        entries.extend([(report['model'],'before',parent),(report['model'],'after',weights)])
    entries.extend([('firebc','baseline',pathlib.Path(baseline['model_path'])),('rules','baseline',None)])
    evaluations=[]
    for name,label,source in entries:
        folder=a.out/(name+'-'+label);folder.mkdir();path=folder/'weights.json' if source else None
        if source:
            model=read(source);model['deterministic']=True;model['sampling_seed']=0;save(path,model)
        plan=folder/'plan.json';root=folder/'capture'
        command=[str(a.compiler.resolve()),'--registry',baseline['registry_path'],'--episodes',','.join(families),'--split','validation','--mode','learned' if source else 'rules','--count','4','--seed-offset','24','--root',str(pathlib.Path(__file__).resolve().parents[1]),'--out',str(plan.resolve()),'--artifacts',str(root.resolve())]
        if path:command+=['--model',str(path.resolve())]
        subprocess.run(command,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),stdout=subprocess.DEVNULL)
        plans.append(str(plan.resolve()));evaluations.append(dict(model=name,label=label,root=str(root.resolve()),plan=str(plan.resolve()),plan_sha256=sha(plan),source_weights_sha256=sha(source) if source else None,deterministic_weights_sha256=sha(path) if path else None))
    save(a.out/'cuda-audit.json',dict(state='passed',device='cuda',checks=checks))
    save(a.out/'plans.json',plans)
    save(a.out/'protocol.json',dict(version='combat_target_bootstrap_paired_validation_v1',evaluations=evaluations,families=families,episodes_per_model=16,total_episodes=288,slots=16,timescale=2,seed_offset=24,split='validation',comparison_reference='firebc-baseline',validation_reused_for_tuning=True,final_test_deferred=True,training_report_sha256=sha(a.models/'target-bc-report.json'),scope='Eight target-head variants before/after equal 50-epoch CUDA bootstrap; four pilot families, fixed conditions and reused validation. Nearest-target heuristic, observed constant-velocity eye-origin blaster labels; machinegun recoil labels excluded. No architecture or final-test superiority claim.'))
    print('Prepared 288 paired native battles',flush=True)

if __name__=='__main__':main()
