"""Export the accepted pooled predictor, preserving separate source manifests."""
import argparse
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('probe','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();assert not args.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path);assert expected is None or expected==digest,str(path)
        pins[str(path)]=digest;return path
    seal=read(pin(args.probe/'complete.json'));report=read(pin(args.probe/'report.json',seal['report_sha256']))
    assert report['selected_regime']=='pooled' and not report['runtime_integrated'] and report['device']=='cuda'
    for fold in report['splits'].values():
        old=fold['legacy_only']['metrics'];new=fold['pooled']['metrics']
        assert new['new_intervened_held_out']['log_loss']<old['new_intervened_held_out']['log_loss']
        assert new['old_held_out']['log_loss']<=old['old_held_out']['log_loss']+.02
    protocol=read(pin(args.probe/'protocol.json'));registry=read(pin(protocol['registry']))
    assert pins[str(Path(protocol['registry']).resolve())]==report['inputs'][str(Path(protocol['registry']).resolve())]
    folders=[Path(protocol['legacy_data'])]+[Path(e['dataset']) for e in registry['entries']];sources=[];source_cases=0
    for folder in folders:
        closed=read(pin(folder/'complete.json'));meta=read(pin(folder/'report.json',closed['report_sha256']))
        pin(folder/'dataset.pt',closed['dataset_sha256']);assert not closed['ppo_ready']
        assert closed['dataset_sha256']==meta['dataset_sha256']
        sources.append(dict(path=str(folder/'dataset.pt'),sha256=closed['dataset_sha256'],shots=meta['shots']))
        source_cases+=meta['source_cases']
    from ppo_combat import torch,nn,training_devices
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from combat_action_geometry import SMOOTH_VERSION,names_for_version,features_for_version
    pin(Path(__file__).with_name('combat_action_geometry.py'))
    tensors=[torch.load(folder/'dataset.pt',map_location='cuda',weights_only=True) for folder in folders]
    for t in tensors:
        assert t['geometry_version']==SMOOTH_VERSION and torch.equal(features_for_version(t['features_with_recoil'],SMOOTH_VERSION),t['action_geometry'])
    x=torch.cat([torch.cat((t['features_with_recoil'],t['action_geometry']),-1) for t in tensors]);y=torch.cat([t['labels'] for t in tensors])
    mean=x.mean(0);std=x.std(0,unbiased=False);std=torch.where(std<1e-3,torch.ones_like(std),std)
    values=(x-mean)/std;torch.manual_seed(protocol['seed']);net=nn.Linear(214,1,device='cuda')
    optimizer=torch.optim.Adam(net.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
    for _ in range(protocol['steps']):
        loss=nn.functional.binary_cross_entropy_with_logits(net(values)[:,0],y)
        assert bool(torch.isfinite(loss));optimizer.zero_grad();loss.backward();optimizer.step()
    args.out.mkdir(parents=True)
    save(args.out/'training-datasets.json',dict(version='combat_predictive_composite_training_v1',purpose='TRAIN_only',sources=sources,regime='pooled',total_shots=len(y)))
    model=dict(kind='combat_geometry_shot_predictor_v1',geometry_version=SMOOTH_VERSION,input_width=214,causal_width=185,
        geometry_names=list(names_for_version(SMOOTH_VERSION)),mean=mean.tolist(),std=std.tolist(),weight=net.weight[0].detach().tolist(),bias=float(net.bias[0].detach()),
        trained_weapon='machinegun',dataset_sha256=sha(args.out/'training-datasets.json'),
        scope='P(live monster damage | actual MG shot, causal observation/action). dataset_sha256 binds composite TRAIN manifest. Pooled old and randomized-aim experience; not optimal-fire or win value.')
    save(args.out/'predictor.json',model);loaded=read(args.out/'predictor.json')
    with torch.no_grad():
        decoded=nn.functional.linear((x-torch.tensor(loaded['mean'],device='cuda'))/torch.tensor(loaded['std'],device='cuda'),
            torch.tensor(loaded['weight'],device='cuda')[None],torch.tensor([loaded['bias']],device='cuda'))[:,0]
        raw=net(values)[:,0];error=float((decoded-raw).abs().max());assert error<1e-5
        train_loss=float(nn.functional.binary_cross_entropy_with_logits(raw,y))
    torch.save(dict(version='combat_geometry_predictor_checkpoint_v1',model=net.state_dict(),optimizer=optimizer.state_dict(),mean=mean,std=std,
        steps=protocol['steps'],training_manifest_sha256=sha(args.out/'training-datasets.json')),args.out/'checkpoint.pt')
    for path,digest in pins.items():assert sha(Path(path))==digest,path
    result=dict(state='complete',version='combat_geometry_predictor_export_v1',device='cuda',parameters=215,source_cases=source_cases,train_shots=len(y),
        regime='pooled',steps=protocol['steps'],train_log_loss=train_loss,export_max_logit_error=error,inputs=pins,script_sha256=sha(__file__),
        predictor_sha256=sha(args.out/'predictor.json'),checkpoint_sha256=sha(args.out/'checkpoint.pt'),
        policy_training=False,runtime_integrated=False,promotion=False,
        scope='Accepted development algorithm refit on all TRAIN sources. Composite manifest, not duplicated raw dataset. Battle policy/default unchanged; fresh native evaluation required.')
    save(args.out/'report.json',result);save(args.out/'complete.json',dict(state='complete',report_sha256=sha(args.out/'report.json'),
        predictor_sha256=result['predictor_sha256'],checkpoint_sha256=result['checkpoint_sha256'],runtime_integrated=False,promotion=False))
    print('Pooled predictor exported on CUDA:',len(y),'TRAIN shots;',source_cases,'source cases')


if __name__=='__main__':main()
