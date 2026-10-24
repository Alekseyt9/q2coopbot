"""Report forecast support and short/long challenge errors without refitting."""
import argparse
from pathlib import Path
import numpy as np
from evaluate_registered_combat_models import read,save,sha
from train_combat_local_effect_predictor_cuda import torch,training_devices,Forecaster,encode,prediction,targets


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','dataset','protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or h==expected,str(path);pins[str(path)]=h;return path
    p=read(pin(a.protocol));ec=read(pin(a.experiment/'complete.json'));er=read(pin(a.experiment/'report.json',ec['report_sha256']))
    ds_closed=read(pin(a.dataset/'complete.json'));ds=read(pin(a.dataset/'report.json',ds_closed['report_sha256']))
    cr=Path(p['challenge']);cc=read(pin(cr/'complete.json'));cm=read(pin(cr/'report.json',cc['report_sha256']))
    roots=read(pin(cr/'records.json',cm['artifacts']['records']))
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    data={}
    for split in ('train','challenge'):
        chunks=[]
        for item in ds['chunks']:
            if item['split']==split:
                with np.load(pin(item['path'],item['sha256']),allow_pickle=False) as f:chunks.append({k:f[k].copy() for k in f.files})
        data[split]={k:torch.as_tensor(np.concatenate([c[k] for c in chunks]),device='cuda') for k in chunks[0]}
    d=data['challenge'];x=encode(d);actual=targets(d)[1]
    train_remaining=torch.unique(data['train']['option'][:,2]);challenge_remaining=torch.unique(d['option'][:,2])
    index={(r['context_id'],r['stream_seed'],r['duration'],r['label']):i for i,r in enumerate(roots)}
    contexts=sorted({r['context_id'] for r in roots});seeds=sorted({r['stream_seed'] for r in roots});results={}
    for name in ('state_action','state_only'):
        checkpoint=pin(a.experiment/name/'checkpoint.pt',er['results'][name]['checkpoint_sha256'])
        model=Forecaster();model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['state_dict']);model.eval()
        inputs=x if name=='state_action' else torch.cat((x[:,:1147],torch.zeros_like(x[:,1147:])),1)
        with torch.no_grad():_,pred=prediction(model(inputs))
        rows=[]
        for duration in (1,5):
            for hi,h in enumerate((1,3,5,10)):
                pp=[];yy=[]
                for cid in contexts:
                    for label in ('left','right','forward','backward'):
                        li=torch.tensor([index[cid,seed,duration,label] for seed in seeds],device='cuda')
                        ri=torch.tensor([index[cid,seed,duration,'control'] for seed in seeds],device='cuda')
                        assert bool(d['known'][li,hi].all() and d['known'][ri,hi].all())
                        pp.append((pred[li,hi]-pred[ri,hi]).mean(0));yy.append((actual[li,hi]-actual[ri,hi]).mean(0))
                pp,yy=torch.stack(pp),torch.stack(yy);rmse=(pp-yy).square().mean(0).sqrt();zero=yy.square().mean(0).sqrt()
                rows.append(dict(duration=duration,horizon=h,contexts=4,mean_action_pairs=len(pp),
                    components={k:dict(rmse=float(rmse[i]),zero_effect_rmse=float(zero[i]),ratio=float(rmse[i]/zero[i]) if float(zero[i])>0 else None) for i,k in enumerate(('received','dealt'))}))
        results[name]=rows
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True);report=dict(state='complete',version='combat_effect_forecast_support_audit_v1',device='cuda',
        train_remaining_frames=train_remaining.tolist(),challenge_remaining_frames=challenge_remaining.tolist(),
        long_duration_outside_training_support=bool(challenge_remaining.max()>train_remaining.max()),results=results,inputs=pins,
        training_performed=False,new_games=False,admission_changed=False,promotion=False,
        scope='Post-hoc CUDA support diagnostic. Long window duration absent from fitting, so its negative challenge alone does not reject an in-support short-action forecaster. Original experiment gates retained; no retuning or promotion.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print({k:report[k] for k in ('train_remaining_frames','challenge_remaining_frames','long_duration_outside_training_support','results')})


if __name__=='__main__':main()
