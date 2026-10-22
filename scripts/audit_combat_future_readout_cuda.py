"""Audit matched readout budgets, causal cache and TRAIN-only normalization."""
import argparse
import pathlib

from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',action='append',type=pathlib.Path,required=True)
    ap.add_argument('--cache',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    args=ap.parse_args();assert len(args.run)==4 and not args.out.exists()
    import numpy as np
    import torch
    assert torch.cuda.is_available(),'CUDA required; no CPU fallback'
    pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path).resolve();key=str(path)
        digest=pins[key] if key in pins else sha(path)
        assert expected is None or digest==expected,key;pins[key]=digest;return path
    c=read(pin(args.cache/'complete.json'));assert c['state']=='complete' and not c['actor_changed']
    assert c['memory_max_error']<2e-5 and c['causal_max_error']<2e-5 and c['head_decomposition_max_error']<1e-6
    assert len(c['cases'])==48 and sum(p['retained'] for p in c['cases'])==5276
    for item in c['inputs']:pin(item['path'],item['sha256'])
    cache=torch.load(pin(args.cache/'cache.pt',c['cache_sha256']),map_location='cuda',weights_only=True)
    activation_stats={}
    for name in ('encoder','attention'):
        values=cache['train'][name];assert bool(torch.isfinite(values).all())
        std=values.std(0)
        activation_stats[name]=dict(nonconstant_channels=int((std>1e-4).sum()),channels=values.shape[1],
                                    mean_channel_std=float(std.mean()),max_channel_std=float(std.max()))
    counts=dict(raw=168739,encoder=30115,attention=30115,combined=38307)
    reports=[];draws={};seen=set()
    for root in args.run:
        closed=read(pin(root/'complete.json'));assert closed['state']=='complete' and closed['device']=='cuda'
        for f,k in (('report.json','report_sha256'),('protocol.json','protocol_sha256'),('checkpoint.pt','checkpoint_sha256')):pin(root/f,closed[k])
        report=read(root/'report.json');p=read(root/'protocol.json');assert report['parameters']==counts and not report['actor_changed']
        assert p['representation_cache_sha256']==sha(args.cache/'complete.json') and p['source_actor_frozen']
        assert p['source_weights_sha256']==c['source_weights_sha256'] and p['outputs']==35
        assert p['steps']==2000 and p['batch_size']==256 and p['learning_rate']==.001 and p['clear_bce_weight']==0
        assert p['identical_encoder_attention_initial_weights'] and p['identical_train_minibatches'] and not p['selection_on_development']
        assert p['common_second_layer_and_action_weights'] and p['clear_metrics_baseline_only']
        for item in p['inputs_sha256']:pin(item['path'],item['sha256'])
        data=pathlib.Path(p['dataset']);assert sha(data/'complete.json')==c['dataset_sha256']
        cp=torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=True)
        assert cp['dataset_sha256']==c['dataset_sha256'] and cp['steps']==2000 and cp['seed']==p['seed']
        for name,state in cp['models'].items():
            assert sum(v.numel() for v in state.values())==counts[name]
            assert state['4.weight'].shape==(35,128) and bool(torch.isfinite(state['4.weight']).all())
        mode=p.get('normalization','none');assert mode in ('none','train_mean_std_floor0.05_excluding_action')
        key=(p['seed'],mode);assert key not in seen;seen.add(key)
        if p['seed'] in draws:assert torch.equal(cp['train_draws'],draws[p['seed']])
        else:draws[p['seed']]=cp['train_draws']
        if mode!='none':
            with np.load(data/'train.npz',allow_pickle=False) as arrays:x=torch.as_tensor(arrays['x'],device='cuda')
            action=x[:,-6:];e,a=cache['train']['encoder'],cache['train']['attention']
            representations=dict(raw=x,encoder=torch.cat((e,action),-1),attention=torch.cat((a,action),-1),combined=torch.cat((e,a,action),-1))
            for name,value in representations.items():
                assert torch.equal(cp['normalization'][name]['mean'],value[:,:-6].mean(0))
                assert torch.equal(cp['normalization'][name]['scale'],value[:,:-6].std(0,unbiased=False).clamp_min(.05))
        for split in ('train','development'):
            for name in counts:
                for metric in ('self_rmse','enemy_rmse'):
                    assert report['before'][split][name]['pooled'][metric]==report['before'][split]['cv']['pooled'][metric]
        methods=report['after']['development'];records={}
        for name in (*counts,'cv'):
            m=methods[name];records[name]=dict(self_rmse=m['pooled']['self_rmse'],enemy_rmse=m['pooled']['enemy_rmse'],
                case_weighted_enemy_rmse=m['case_weighted']['enemy_rmse']['rmse'],
                track_brier=m['pooled']['track_present']['brier'],maps={k:v['enemy_rmse'] for k,v in m['maps'].items()})
        reports.append(dict(root=str(root.resolve()),seed=p['seed'],normalization=mode,methods=records,complete_sha256=sha(root/'complete.json')))
    assert len(draws)==2 and all((s,m) in seen for s in draws for m in ('none','train_mean_std_floor0.05_excluding_action'))
    source=pathlib.Path(c['source']);assert sha(source/'weights.json')==c['source_weights_sha256'] and sha(source/'checkpoint.pt')==c['source_checkpoint_sha256']
    for path,digest in pins.items():assert sha(path)==digest
    save(args.out,dict(state='complete',version='combat_frozen_future_readout_audit_v1',device='cuda',runs=reports,
        parameter_counts=counts,activation_stats=activation_stats,matched_labels_and_minibatches=True,normalization_train_only=True,
        causal_cache_verified=True,source_actor_unchanged=True,training_performed=False,promotion=False,
        decision='frozen_trunk_readout_weak_for_enemy_forecast_auxiliary_utility_still_unproven',
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),
        scope='Two seeds and a normalization control on the same16 held-out cases. Residual decoder uses explicit observed CV bypass in every arm. Different decoder input sizes/capacities; source spatial/weapon branches also consume raw features. Probe failure is not proof of absent information in the whole policy. No policy update or game-quality result.'))
    print(args.out)


if __name__=='__main__':main()
