"""Continue a sealed policy on one closed objective-matched train corpus."""
import argparse
import json
import pathlib
import subprocess
import sys
from process_combat_architecture_pool import read, save, sha


def main():
    ap=argparse.ArgumentParser()
    for name in ('source','corpus','anchor','bank','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--sequence-retention-weight',type=float,default=1.)
    args=ap.parse_args();assert not args.out.exists() and 0<=args.sequence_retention_weight<=10
    import torch
    assert torch.cuda.is_available(), 'CUDA required; no CPU fallback'
    repo=pathlib.Path(__file__).resolve().parents[1]
    source,corpus,out=[p.resolve() for p in (args.source,args.corpus,args.out)]
    seal=read(source/'complete.json');pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path);digest=sha(path);assert expected is None or digest==expected
        if str(path) in pins:assert pins[str(path)]==digest
        pins[str(path)]=digest;return path
    pin(source/'complete.json')
    for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:pin(source/name,seal[key])
    assert read(source/'report.json')['device']=='cuda'
    cp=torch.load(source/'checkpoint.pt',map_location='cuda',weights_only=True)
    assert cp['weights_sha256']==seal['weights_sha256']
    pin(args.anchor,cp['anchor_sha256']);pin(args.bank,cp['bank_sha256'])
    closed=read(pin(corpus/'complete.json'));assert closed['state']=='complete' and not closed['training_performed']
    protocol=read(pin(corpus/'protocol.json',closed['protocol_sha256']))
    assert protocol['source']==str(source) and protocol['train_only'] and not protocol['diagnostic_intervention']
    objective=protocol['ppo_training_objective'];assert objective
    assert objective['checkpoint_sha256']==seal['checkpoint_sha256']
    config=pin(objective['config'],objective['config_sha256'])
    assert read(config)==cp['config'] and objective['reward_sha256']==cp['config']['objective_reward_sha256']
    for binding in protocol['inputs']:pin(binding['path'],binding['sha256'])
    assert len(closed['processed'])==1 and closed['processed'][0]['label']=='train'
    binding=closed['processed'][0];processing=pathlib.Path(binding['report']).parent
    processed=read(pin(processing/'report.json',binding['sha256']))
    assert processed['state']=='complete' and processed['device']=='cuda' and not processed['training_performed']
    assert all(c['split']=='train' for c in processed['cases'])
    merged=read(pin(processing/'merged/report.json'))
    assert merged['model_sha256']==seal['weights_sha256'] and merged['reward_config_sha256']==objective['reward_sha256']
    for name in ('rollout','sequence'):pin(processing/'merged'/(name+'.jsonl'),merged[name+'_sha256'])
    weapon_lr=cp.get('actor_training',{}).get('weapon_state_residual',{}).get('learning_rate',.0003)
    del cp
    out.mkdir(parents=True)
    def run(label,arguments):
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            completed=subprocess.run([sys.executable,*map(str,arguments)],cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert completed.returncode==0,str(out/(label+'.log'))
    try:
        save(out/'progress.json',dict(stage='preparing-train-retention',training_cases=len(processed['cases'])))
        spec=out/'sequence-retention.json'
        run('retention',[repo/'scripts/prepare_combat_sequence_retention.py','--pool',corpus/'pool',
            '--plan-index','0','--processing',processing,'--out',spec])
        pin(spec)
        save(out/'progress.json',dict(stage='cuda-training',training_rows=merged['rows']))
        update=out/'update'
        run('train',[repo/'scripts/ppo_recurrent.py','--model',source/'weights.json','--data',processing/'merged',
            '--config',config,'--resume',source/'checkpoint.pt','--anchor-model',args.anchor.resolve(),
            '--retention-bank',args.bank.resolve(),'--retention-weight','0','--bank-weight','0',
            '--sequence-retention-spec',spec,'--sequence-retention-weight',args.sequence_retention_weight,
            '--fork-sequence-retention','--weapon-state-lr',weapon_lr,'--out',update])
        result=read(update/'report.json');receipt=read(update/'complete.json')
        assert result['device']=='cuda' and result['behavior_sha256']==seal['weights_sha256']
        for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:pin(update/name,receipt[key])
        assert result['rollout_sha256']==merged['rollout_sha256']
        for path,digest in pins.items():assert sha(path)==digest
        save(out/'complete.json',dict(state='complete',version='registered_ppo_fit_v1',device='cuda',
            source=str(source),source_weights_sha256=seal['weights_sha256'],corpus=str(corpus),update=str(update),
            training_cases=len(processed['cases']),rows=merged['rows'],accepted_actor_steps=result['actor_steps'],
            policy_updated=result['actor_steps']>0,update_complete_sha256=sha(update/'complete.json'),
            inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),
            training_performed=True,promotion=False,
            scope='Fresh objective-matched train-only full-history CUDA PPO; frozen initial behavior retention, all available target/mode branches, actual recorded movement conditioning. No validation states fit, no model promotion.'))
        save(out/'progress.json',dict(stage='complete',accepted_actor_steps=result['actor_steps']))
        print(json.dumps(dict(state='complete',update=str(update),accepted_actor_steps=result['actor_steps'],rows=merged['rows'])))
    except BaseException as error:
        save(out/'progress.json',dict(stage='failed',error=str(error)));raise


if __name__=='__main__':main()
