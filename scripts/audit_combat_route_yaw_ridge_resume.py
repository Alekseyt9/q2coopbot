"""Actual split/append CUDA fits must reproduce the one-shot yaw ridge model."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
from ppo_combat import torch
from combat_checkpoint_scope import require_checkpoint_scope
from train_combat_route_yaw_ridge import SCOPE,identity
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True,type=Path);ap.add_argument('--data',required=True,type=Path)
    ap.add_argument('--full-fit',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--movement-given-yaw',action='store_true',help='Audit conditional two-row movement ridge instead of yaw row')
    a=ap.parse_args();assert not a.out.exists();a.out.mkdir(parents=True)
    data=read(a.data);train=[e for e in data['episodes'] if e['split']=='train'];diagnostic=[e for e in data['episodes'] if e['split']=='diagnostic']
    halves=(train[:len(train)//2],train[len(train)//2:]);assert all(halves)
    for name,episodes in zip(('first','next'),halves):
        subset=copy.deepcopy(data);subset['episodes']=[*episodes,*diagnostic];subset['subset_of_sha256']=sha(a.data)
        subset['counts']['train']=dict(rows=sum(e['rows'] for e in episodes),worlds=len(episodes))
        save(a.out/(name+'-data.json'),subset)
    trainer=Path(__file__).with_name('train_combat_route_movement_given_yaw.py' if a.movement_given_yaw else 'train_combat_route_yaw_ridge.py')
    def run(name,model,data,out,resume=None,fail=False):
        cmd=[sys.executable,str(trainer),'--model',str(model),'--data',str(data),'--out',str(out)]
        if resume:cmd+=['--resume',str(resume)]
        with (a.out/(name+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        if fail:
            assert r.returncode!=0 and 'Data already consumed' in (a.out/(name+'.log')).read_text() and not out.exists()
        else:assert r.returncode==0,str(a.out/(name+'.log'))
    run('first',a.model,a.out/'first-data.json',a.out/'first-fit')
    run('append',a.out/'first-fit/weights.json',a.out/'next-data.json',a.out/'append-fit',a.out/'first-fit/checkpoint.pt')
    torch.set_default_device('cuda');assert torch.cuda.is_available()
    full=read(a.full_fit/'weights.json');appended=read(a.out/'append-fit/weights.json')
    if a.movement_given_yaw:
        from train_combat_route_movement_given_yaw import frozen,SCOPE as expected_scope
        assert frozen(full)==frozen(appended)
        row=lambda m:[v for i in (0,1) for v in [*m['memory_route_movement']['layers'][-1]['weight'][i],m['memory_route_movement']['layers'][-1]['bias'][i]]]
    else:
        expected_scope=SCOPE
        assert identity(full)==identity(appended)
        row=lambda m:[*m['memory_route_movement']['layers'][-1]['weight'][2],m['memory_route_movement']['layers'][-1]['bias'][2]]
    left=torch.tensor(row(full),device='cuda',dtype=torch.float64);right=torch.tensor(row(appended),device='cuda',dtype=torch.float64)
    torch.testing.assert_close(left,right,atol=1e-5,rtol=1e-5)
    cp=torch.load(a.full_fit/'checkpoint.pt',map_location='cuda',weights_only=True)
    resumed=torch.load(a.out/'append-fit/checkpoint.pt',map_location='cuda',weights_only=True)
    assert cp['rows']==resumed['rows'] and cp['consumed']==resumed['consumed'] and cp['worlds']==resumed['worlds']
    torch.testing.assert_close(cp['gram'],resumed['gram'],atol=1e-7,rtol=1e-12)
    torch.testing.assert_close(cp['rhs'],resumed['rhs'],atol=1e-7,rtol=1e-12)
    try:require_checkpoint_scope(cp)
    except ValueError:pass
    else:raise AssertionError('Dedicated ridge state admitted to PPO')
    run('duplicate-rejection',a.out/'append-fit/weights.json',a.out/'next-data.json',a.out/'must-not-exist',a.out/'append-fit/checkpoint.pt',fail=True)
    save(a.out/'report.json',dict(state='passed',device='cuda',version='combat_yaw_ridge_append_resume_audit_v1',
        rows=cp['rows'],worlds=len(cp['worlds']),weights_max_difference=float((left-right).abs().max()),
        actual_cli_append_verified=True,sufficient_statistics_verified=True,duplicate_data_rejected=True,
        generic_ppo_rejected=True,checkpoint_training_scope=expected_scope,gameplay_quality_claim=False))
    print('CUDA ridge split/append and duplicate rejection passed',flush=True)


if __name__=='__main__':main()
