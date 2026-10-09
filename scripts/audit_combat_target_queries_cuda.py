"""Check serialized observed queries and feature-derived labels on CUDA only."""
import argparse,gzip,json,pathlib
from process_combat_architecture_pool import read,sha,save
from train_combat_target_bc import torch,annotations
from combat_target_head import availability

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--reference-data',type=pathlib.Path);a=ap.parse_args()
    assert torch.cuda.is_available();meta=read(a.data/'report.json');assert meta['version']=='combat_target_sequence_v1'
    reports=[];seeds={}
    for split in ('train','validation'):
        path=a.data/(split+'.jsonl.gz');assert sha(path)==meta['data_sha256'][split]
        with gzip.open(path,'rt',encoding='utf-8') as stream:rows=[json.loads(line) for line in stream]
        assert len(rows)==meta['counts'][split];seeds[split]={r['seed'] for r in rows}
        x=torch.tensor([r['features'] for r in rows],device='cuda');query_move=torch.tensor([r.get('query_move_label',[0,0,0]) for r in rows],device='cuda');query_ducked=torch.tensor([r.get('query_ducked_label',bool(r['features'][3])) for r in rows],device='cuda')
        angles,mask,target=annotations(x,meta.get('target_query_version','observed_target_aim_query_v1'),query_move,query_ducked)
        prior=x[:,845:854];assert bool(((prior==0)|(prior==1)).all()) and bool((prior.sum(1)==1).all())
        indices=[];expected=[];lead_masks=[]
        for i,row in enumerate(rows):
            if row['target_queries'] is None:assert not row['mask'][1]
            for label in row['target_queries'] or []:
                slot=label['slot']-1;assert 0<=slot<8
                indices.append((i,slot));expected.append([label['yaw_delta_degrees'],label['pitch_delta_degrees']]);lead_masks.append(label['lead_known'] and label['recoil_known'])
        index=torch.tensor(indices,dtype=torch.long,device='cuda');expected=torch.tensor(expected,device='cuda')
        expected_mask=torch.tensor(lead_masks,dtype=torch.bool,device='cuda')
        actual_mask=mask[index[:,0],index[:,1]];assert torch.equal(actual_mask,expected_mask),'Serialized query/feature lead masks differ'
        difference=(angles[index[:,0],index[:,1]]*180-expected+180).remainder(360)-180
        error=float(difference[actual_mask].abs().max());assert error<.03,('Observed interception query mismatch',split,error)
        prior_slot=prior.argmax(1);available=availability(x).gather(1,prior_slot[:,None])[:,0]
        retained=(prior_slot>0)&available
        assert bool((target[retained]==prior_slot[retained]).all())
        if a.reference_data:
            reference_meta=read(a.reference_data/'report.json');reference_path=a.reference_data/(split+'.jsonl.gz');assert sha(reference_path)==reference_meta['data_sha256'][split]
            with gzip.open(reference_path,'rt',encoding='utf-8') as f:reference=[json.loads(s) for s in f]
            assert [r['identity'] for r in rows]==[r['identity'] for r in reference] and [r['seed'] for r in rows]==[r['seed'] for r in reference]
            assert torch.equal(x,torch.tensor([r['features'] for r in reference],device='cuda')),'Future labels changed policy features'
        reports.append(dict(split=split,frames=len(rows),known_aim_pairs=int(actual_mask.sum()),retained_target_frames=int(retained.sum()),maximum_query_error_degrees=error,missing_postmove_context=sum('query_move_label' not in r for r in rows) if meta.get('target_query_version')=='observed_center_blaster_postmove_query_v3' else 0,reference_features_exact=True if a.reference_data else None))
    assert not seeds['train'] & seeds['validation']
    save(a.out,dict(state='passed',device='cuda',data_report_sha256=sha(a.data/'report.json'),splits=reports,scope='Client feature/query geometry, lead masks, identity-remapped intent and split separation. No Go model inference, shot-hit claim or quality promotion.'))
    print(json.dumps(reports),flush=True)

if __name__=='__main__':main()
