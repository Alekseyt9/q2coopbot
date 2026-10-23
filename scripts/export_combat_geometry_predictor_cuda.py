"""Refit the accepted linear214 predictor on pooled TRAIN and export its weights.

Exports a predictor, not a battle policy or a firing gate. Live integration and
fresh native evaluation remain separate acceptance requirements.
"""
import argparse
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('probe','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();assert not args.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path);assert expected is None or digest==expected,str(path)
        pins[str(path)]=digest;return path
    seal=read(pin(args.probe/'complete.json'));report=read(pin(args.probe/'report.json',seal['report_sha256']))
    assert report['state']=='complete' and report['device']=='cuda' and not report['runtime_integrated']
    protocol=read(pin(args.probe/'protocol.json'))
    for fold in report['splits'].values():
        candidate=fold['models']['linear214']
        for baseline in ('linear181','linear185'):
            for subset in ('held_out','held_out_selected_target'):
                assert candidate[subset]['log_loss']<fold['models'][baseline][subset]['log_loss']
        for direction in ('away','toward'):
            assert fold['action_sensitivity']['linear214'][direction]['fraction_expected_sign']>.9
    data=Path(protocol['data']);data_seal=read(pin(data/'complete.json'))
    data_report=read(pin(data/'report.json',data_seal['report_sha256']))
    pin(data/'dataset.pt',data_report['dataset_sha256'])
    assert pins[str((data/'dataset.pt').resolve())]==report['inputs'][str((data/'dataset.pt').resolve())]
    assert not data_seal['ppo_ready'] and data_report['source_cases']==288
    from ppo_combat import torch,nn,training_devices
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from combat_action_geometry import names_for_version,features_for_version
    pin(Path(__file__).with_name('combat_action_geometry.py'))
    tensors=torch.load(data/'dataset.pt',map_location='cuda',weights_only=True)
    VERSION=tensors['geometry_version'];NAMES=names_for_version(VERSION);features=lambda x:features_for_version(x,VERSION)
    assert tensors['geometry_names']==list(NAMES) and torch.equal(features(tensors['features_with_recoil']),tensors['action_geometry'])
    x=torch.cat((tensors['features_with_recoil'],tensors['action_geometry']),-1);y=tensors['labels']
    assert x.shape==(9248,214)
    mean=x.mean(0);std=x.std(0,unbiased=False);std=torch.where(std<1e-3,torch.ones_like(std),std)
    values=(x-mean)/std;torch.manual_seed(protocol['seed']);net=nn.Linear(214,1,device='cuda')
    optimizer=torch.optim.Adam(net.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
    for _ in range(protocol['steps']):
        loss=nn.functional.binary_cross_entropy_with_logits(net(values)[:,0],y)
        assert bool(torch.isfinite(loss));optimizer.zero_grad();loss.backward();optimizer.step()
    model=dict(kind='combat_geometry_shot_predictor_v1',geometry_version=VERSION,input_width=214,causal_width=185,
        geometry_names=list(NAMES),mean=mean.tolist(),std=std.tolist(),weight=net.weight[0].detach().tolist(),bias=float(net.bias[0].detach()),
        trained_weapon='machinegun',dataset_sha256=data_report['dataset_sha256'],
        scope='P(live monster health damage | an actual MG shot, causal observation, chosen action); not P(optimal fire), not a win-value model')
    args.out.mkdir(parents=True);save(args.out/'predictor.json',model)
    exported=read(args.out/'predictor.json')
    with torch.no_grad():
        reconstructed=nn.functional.linear((x-torch.tensor(exported['mean'],device='cuda'))/torch.tensor(exported['std'],device='cuda'),
            torch.tensor(exported['weight'],device='cuda')[None],torch.tensor([exported['bias']],device='cuda'))[:,0]
        raw=net(values)[:,0];error=float((raw-reconstructed).abs().max());assert error<1e-5
        probability=raw.sigmoid();train_loss=float(nn.functional.binary_cross_entropy_with_logits(raw,y))
        train_brier=float((probability-y).square().mean())
    torch.save(dict(version='combat_geometry_predictor_checkpoint_v1',model=net.state_dict(),optimizer=optimizer.state_dict(),
        mean=mean,std=std,steps=protocol['steps'],dataset_sha256=data_report['dataset_sha256']),args.out/'checkpoint.pt')
    for path,digest in pins.items():assert sha(Path(path))==digest,path
    receipt=dict(state='complete',version='combat_geometry_predictor_export_v1',device='cuda',train_shots=len(y),parameters=215,
        steps=protocol['steps'],source_cases=data_report['source_cases'],train_log_loss=train_loss,train_brier=train_brier,
        export_max_logit_error=error,inputs=pins,script_sha256=sha(__file__),
        predictor_sha256=sha(args.out/'predictor.json'),checkpoint_sha256=sha(args.out/'checkpoint.pt'),
        policy_training=False,runtime_integrated=False,promotion=False,
        scope='Refit chosen algorithm on all pooled TRAIN shots after grouped development comparisons. No validation/test cases, battle-policy tensors, source Adam/RNG or default weights changed. A learned aim-ranking prototype requires Go feature parity and actual native evaluation before any use.')
    save(args.out/'report.json',receipt)
    save(args.out/'complete.json',dict(state='complete',report_sha256=sha(args.out/'report.json'),predictor_sha256=receipt['predictor_sha256'],
        checkpoint_sha256=receipt['checkpoint_sha256'],runtime_integrated=False,promotion=False))
    print(json.dumps({k:receipt[k] for k in ('train_shots','parameters','train_log_loss','export_max_logit_error','runtime_integrated')}))


if __name__=='__main__':main()
