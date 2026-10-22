"""Close the two-seed predictive pilot; no additional fitting or policy changes."""
import argparse
import pathlib

from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--first',type=pathlib.Path,required=True)
    ap.add_argument('--replica',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    args=ap.parse_args();assert not args.out.exists()
    import numpy as np
    import torch
    assert torch.cuda.is_available(),'CUDA required; no CPU fallback'
    receipts=[];records=[];datasets=[];protocols=[]
    for root in (args.first,args.replica):
        complete=read(root/'complete.json');assert complete['state']=='complete' and complete['device']=='cuda'
        for name,key in (('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256'),('protocol.json','protocol_sha256')):
            assert sha(root/name)==complete[key]
        p=read(root/'protocol.json');report=read(root/'report.json')
        assert p['identical_initial_weights'] and p['identical_train_minibatches'] and not p['selection_on_development']
        assert not report['actor_changed'] and report['steps']==p['steps']==2000 and report['parameters']==169771
        data=pathlib.Path(p['dataset']);meta=read(data/'complete.json');datasets.append((data,meta));protocols.append(p)
        checkpoint=torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=True)
        assert checkpoint['seed']==p['seed'] and checkpoint['dataset_sha256']==sha(data/'complete.json')
        assert checkpoint['train_draws'].shape==(2000,256)
        with np.load(data/'train.npz',allow_pickle=False) as arrays:
            masks=torch.as_tensor(arrays['present_mask'],device='cuda')
        assert bool(masks[checkpoint['train_draws']].any(-1).any(-1).all()), 'Empty-mask guard must be inactive in both fits'
        for split in ('train','development'):
            for metric in ('self_rmse','enemy_rmse'):
                for model in ('action','blind'):
                    assert report['before'][split][model]['pooled'][metric]==report['before'][split]['cv']['pooled'][metric]
        d=report['after']['development'];a=d['action'];cv=d['cv'];blind=d['blind']
        records.append(dict(seed=p['seed'],enemy_rmse=a['pooled']['enemy_rmse'],cv_enemy_rmse=cv['pooled']['enemy_rmse'],
            enemy_improvement_percent=100*(1-a['pooled']['enemy_rmse']/cv['pooled']['enemy_rmse']),
            case_weighted_enemy_rmse=a['case_weighted']['enemy_rmse']['rmse'],
            self_rmse=a['pooled']['self_rmse'],blind_self_rmse=blind['pooled']['self_rmse'],
            action_sensitive_self_rmse=a['pooled']['action_sensitive_self_rmse'],blind_action_sensitive_self_rmse=blind['pooled']['action_sensitive_self_rmse'],
            pmove_common_neural_rmse=a['pooled']['pmove_common_self_rmse'],pmove=d['pmove'],
            enemy_cases_better_than_cv=sum(x['enemy_rmse']<y['enemy_rmse'] for x,y in zip(a['cases'],cv['cases'])),
            all_maps_enemy_better_than_cv=all(v['enemy_rmse']<cv['maps'][k]['enemy_rmse'] for k,v in a['maps'].items()),
            track_brier=a['pooled']['track_present']['brier'],persistence_brier=cv['pooled']['track_present']['brier']))
        receipts.append(dict(root=str(root.resolve()),complete_sha256=sha(root/'complete.json')))
    assert protocols[0]['seed']!=protocols[1]['seed']
    for key in ('steps','batch_size','learning_rate','weight_decay','hidden','inputs','outputs','motion_weight','present_bce_weight','clear_bce_weight'):
        assert protocols[0][key]==protocols[1][key],key
    for split in ('train','development'):
        with np.load(datasets[0][0]/(split+'.npz'),allow_pickle=False) as a,np.load(datasets[1][0]/(split+'.npz'),allow_pickle=False) as b:
            assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
            assert bool((a['clear'][a['clear_mask']]==1).all()), 'Conditional clear head is single-class in this contract'
    gate=all(r['enemy_improvement_percent']>0 and r['all_maps_enemy_better_than_cv']
             and r['action_sensitive_self_rmse']<r['blind_action_sensitive_self_rmse'] for r in records)
    save(args.out,dict(state='complete',version='combat_future_predictive_replication_v1',device='cuda',records=records,
        same_numeric_dataset=True,matched_budget=True,independent_train_seeds=True,source_receipts=receipts,
        predictive_gate_passed=gate,decision='forecast_signal_replicated_actor_utility_unproven' if gate else 'predictive_gate_failed',
        conditional_clear_head_degenerate=True,actor_changed=False,training_performed=False,promotion=False,script_sha256=sha(__file__),
        scope='Two fixed-final supervised fits. Enemy CV gain and action conditioning replicated on16 held-out cases, not new independent maps or policy distributions. Known PMove greatly stronger than neural ego predictor on its supported subset. Clear-shot conditional on retained track is true by policy.Observe contract and is not visibility evidence. No native combat improvement established.'))
    print(args.out)


if __name__=='__main__':main()
