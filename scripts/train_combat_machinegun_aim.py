"""CUDA-only common coarse/fine residual bootstrap on sealed native labels."""
import argparse,copy,gzip,json,pathlib,time
from process_combat_architecture_pool import read,save,sha
from prepare_combat_target_heads import build,forward
from combat_spatial_aim import inputs,export,validate,COARSE_VERSION
from train_combat_sequence_bc import sequences
from train_combat_bc import torch

def prepare(actor,rows):
    groups=sequences(rows);x=torch.zeros(len(groups),max(map(len,groups)),854,device='cuda');indices=[];slots=[];desired=[];weapons=[]
    for i,group in enumerate(groups):
        x[i,:len(group)]=torch.tensor([r['features'] for r in group],device='cuda')
        for j,row in enumerate(group):
            for q in row['supervised_aim_queries']:
                slot=q['slot']-1;assert 0<=slot<8
                indices.append(i*x.shape[1]+j);slots.append(slot);desired.append([q['yaw_delta_degrees'],q['pitch_delta_degrees']]);weapons.append(row['aim_label_weapon']=='machinegun')
    assert indices
    with torch.no_grad():
        base=forward(actor.base,x).reshape(-1,81);feature=x.reshape(-1,854)
        index=torch.tensor(indices,dtype=torch.long,device='cuda');slot=torch.tensor(slots,dtype=torch.long,device='cuda')
        spatial=inputs(feature,base)[index,slot].detach()
        coarse=base[:,29:45].reshape(-1,8,2)[index,slot].detach()
        fine=base[:,47:63].reshape(-1,8,2)[index,slot].detach()
        mode=base[:,65:81].reshape(-1,8,2)[index,slot].detach()
    return spatial,coarse,fine,mode,torch.tensor(desired,device='cuda'),torch.tensor(weapons,dtype=torch.bool,device='cuda')

def loss(branch,data):
    spatial,coarse,fine,mode,desired,machinegun=data;delta=branch(spatial)
    fine_mask=desired.abs().amax(-1)<=15
    assert fine_mask.any() and machinegun.any() and (~machinegun).any()
    coarse_error=((coarse+delta[:,4:6]).tanh()*180-desired+180).remainder(360)-180
    fine_error=(fine+delta[:,:2]).tanh()*15-desired
    ce=torch.nn.functional.cross_entropy(mode+delta[:,2:4],fine_mask.long(),reduction='none')
    # Equal weapon weights keep numerous blaster queries from drowning MG.
    objectives=[]
    for selected in (machinegun,~machinegun):
        term=coarse_error[selected].square().mean()/2025+.05*ce[selected].mean()
        fine_selected=selected&fine_mask
        if fine_selected.any():term=term+fine_error[fine_selected].square().mean()/225
        objectives.append(term)
    objective=sum(objectives)/2
    chosen_error=torch.where(fine_mask[:,None],fine_error,coarse_error)
    metrics=dict(coarse_rmse_degrees=float(coarse_error.detach().square().mean().sqrt()),fine_rmse_degrees=float(fine_error[fine_mask].detach().square().mean().sqrt()),mode_accuracy=float(((mode+delta[:,2:4]).argmax(-1)==fine_mask).float().mean()),
        machinegun_query_rmse_degrees=float(chosen_error[machinegun].detach().square().mean().sqrt()),blaster_query_rmse_degrees=float(chosen_error[~machinegun].detach().square().mean().sqrt()),pairs=len(desired))
    return objective,metrics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',type=pathlib.Path,required=True);ap.add_argument('--data',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--epochs',type=int,default=500);a=ap.parse_args()
    assert torch.cuda.is_available() and 1<=a.epochs<=3000 and not a.out.exists();a.out.mkdir(parents=True)
    torch.set_num_threads(2);torch.manual_seed(20261010);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    migration=read(a.models/'cuda-migration-audit.json');assert migration['state']=='passed' and migration['device']=='cuda'
    meta=read(a.data/'report.json');assert meta['state']=='complete' and meta['version']=='native_conditional_aim_labels_v1' and meta['features_preserved']
    rows={}
    for split in ('train','validation'):
        path=a.data/f'{split}.jsonl.gz';assert sha(path)==meta['data_sha256'][split]
        with gzip.open(path,'rt',encoding='utf-8') as stream:rows[split]=[json.loads(line) for line in stream]
    assert not {r['seed'] for r in rows['train']} & {r['seed'] for r in rows['validation']}
    reports=[]
    for entry in migration['models']:
        source=a.models/entry['model']/'weights.json';assert sha(source)==entry['weights_sha256']
        model=read(source);validate(model);assert model['spatial_aim']['version']==COARSE_VERSION
        actor=build(model,'actor');actor.eval()
        for p in actor.base.parameters():p.requires_grad_(False)
        original={k:v.detach().clone() for k,v in actor.base.state_dict().items()}
        data={split:prepare(actor,values) for split,values in rows.items()}
        with torch.no_grad():before={split:loss(actor.branch,d)[1] for split,d in data.items()}
        opt=torch.optim.Adam(actor.branch.parameters(),lr=.001);started=time.perf_counter()
        for _ in range(a.epochs):
            opt.zero_grad();objective,_=loss(actor.branch,data['train']);assert bool(torch.isfinite(objective));objective.backward();torch.nn.utils.clip_grad_norm_(actor.branch.parameters(),1.);opt.step()
        assert all(torch.equal(value,original[key]) for key,value in actor.base.state_dict().items())
        with torch.no_grad():after={split:loss(actor.branch,d)[1] for split,d in data.items()}
        updated=copy.deepcopy(model);export(updated,actor);validate(updated);assert updated['actor']==model['actor'] and updated['value']==model['value'] and updated['log_std']==model['log_std']
        folder=a.out/entry['model'];folder.mkdir();save(folder/'weights.json',updated)
        rebuilt=build(updated,'actor');assert all(torch.equal(v,rebuilt.state_dict()[k]) for k,v in actor.state_dict().items())
        torch.save(dict(version='native_conditional_aim_bc_v1',actor=actor.state_dict(),optimizer=opt.state_dict(),weights_sha256=sha(folder/'weights.json'),epochs=a.epochs),folder/'checkpoint.pt')
        restored=torch.load(folder/'checkpoint.pt',map_location='cuda',weights_only=True);assert all(torch.equal(v,restored['actor'][k]) for k,v in actor.state_dict().items())
        report=dict(model=entry['model'],architecture=entry['architecture'],device='cuda',epochs=a.epochs,seconds=time.perf_counter()-started,before=before,after=after,parent_sha256=sha(source),weights_sha256=sha(folder/'weights.json'),checkpoint_sha256=sha(folder/'checkpoint.pt'),base_parameters_preserved=True,cuda_export_and_checkpoint_exact=True,data_report_sha256=sha(a.data/'report.json'))
        save(folder/'report.json',report);reports.append(report);save(a.out/'progress.json',dict(stage='training',complete_models=len(reports),total_models=len(migration['models'])))
        print(entry['model'],json.dumps(after),flush=True)
    save(a.out/'report.json',dict(state='complete',device='cuda',gpu=torch.cuda.get_device_name(),models=reports,data_report_sha256=sha(a.data/'report.json'),trainer_sha256=sha(pathlib.Path(__file__)),scope='Equal epochs and identical sealed corpus for every architecture. Only shared coarse/fine/mode residual trained. Native muzzle/recoil occur only in offline labels. New BC Adam optimizer, no inherited PPO state modified. Query fit is not native gameplay quality; no promotion.'))
    save(a.out/'progress.json',dict(stage='complete',models=len(reports),report_sha256=sha(a.out/'report.json')))

if __name__=='__main__':main()
