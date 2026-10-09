"""CUDA intercept-equation residuals and distance-stratified parallax effects."""
import argparse,gzip,json,pathlib
from process_combat_architecture_pool import read,sha,save
from train_combat_target_bc import torch,annotations

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
    assert torch.cuda.is_available();meta=read(a.data/'report.json');assert meta['target_query_version']=='observed_center_blaster_muzzle_query_v2'
    reports=[]
    for split in ('train','validation'):
        path=a.data/(split+'.jsonl.gz');assert sha(path)==meta['data_sha256'][split]
        with gzip.open(path,'rt',encoding='utf-8') as f:rows=[json.loads(s) for s in f]
        x=torch.tensor([r['features'] for r in rows],device='cuda');new,mask,_=annotations(x,meta['target_query_version']);old,old_mask,_=annotations(x)
        n=len(x);enemy=x[:,73:169].reshape(n,8,12);box=x[:,466:786].reshape(n,8,40)[:,:,36:40]
        bottom,top=box[:,:,2]*64,box[:,:,3]*64
        height=torch.where(top-bottom<16,(top+bottom)/2,torch.minimum(torch.maximum(torch.full_like(bottom,22),bottom+8),top-8))
        r=enemy[:,:,2:5]*512;r=r.clone();r[:,:,2]+=height-torch.where(x[:,3:4]==1,-2.,22.)+8
        velocity=enemy[:,:,6:9]*400
        times=torch.zeros_like(mask,dtype=torch.float32)
        for row_index,row in enumerate(rows):
            for label in row['target_queries']:
                if label['lead_known'] and label['recoil_known']:times[row_index,label['slot']-1]=label['lead_seconds']
        yaw=new[:,:,0]*torch.pi;pitch=new[:,:,1]*torch.pi+torch.atan2(x[:,8:9],x[:,9:10])
        d=torch.stack((pitch.cos()*yaw.cos(),pitch.cos()*yaw.sin(),-pitch.sin()),-1)
        residual=(r+velocity*times[:,:,None]-d*(24+1000*times)[:,:,None]).norm(dim=-1)
        maximum=float(residual[mask].max());assert maximum<.02
        difference=((new-old)*180+180).remainder(360)-180
        common=mask&old_mask;distance=enemy[:,:,1]*512;bins={}
        for name,lo,hi in [('near',0,128),('medium',128,512),('far',512,float('inf'))]:
            selected=common&(distance>lo)&(distance<=hi);count=int(selected.sum())
            bins[name]=dict(labels=count,pitch_correction_mean_degrees=float(difference[:,:,1][selected].abs().mean()) if count else None,pitch_correction_max_degrees=float(difference[:,:,1][selected].abs().max()) if count else None,yaw_correction_mean_degrees=float(difference[:,:,0][selected].abs().mean()) if count else None)
        reports.append(dict(split=split,ballistic_residual_max_units=maximum,distance_bins=bins))
    # Exact synthetic stationary targets and explicit inside-muzzle unknown mask.
    distances=torch.tensor([4.,24.,32.,64.,128.,512.,2048.],device='cuda')
    synthetic=torch.zeros(7,854,device='cuda');synthetic[:,9]=1;synthetic[:,17]=1;synthetic[:,426]=1;synthetic[:,78]=1;synthetic[:,75]=distances/512
    # bbox slot0 begins466; bottom/top are504/505.
    synthetic[:,504]=-.375;synthetic[:,505]=1
    fine,known,_=annotations(synthetic,meta['target_query_version']);assert not bool(known[0,0]) and bool(known[1:-1,0].all()) and not bool(known[-1,0])
    expected=torch.atan2(torch.full_like(distances,8),distances)
    assert torch.allclose(fine[1:,0,1]*torch.pi,-expected[1:],atol=1e-6)
    save(a.out,dict(state='passed',device='cuda',data_report_sha256=sha(a.data/'report.json'),splits=reports,stationary_geometry_passed=True,inside_forward_muzzle_masked=True,beyond_projectile_lifetime_masked=True,scope='Instantaneous center-hand/aimfix0 equation and eye-vs-muzzle query difference. Distance bins use observed enemy-origin range. Not firing-phase accuracy, hit rate or live quality.'))
    print(json.dumps(reports),flush=True)

if __name__=='__main__':main()
