"""CUDA forward/replay verification only; never performs an optimizer step."""
import argparse, json, pathlib
from ppo_recurrent import torch, prepare_context, sequence, probabilities, sha
from combat_attention import CausalAttention
from combat_weapon_head import feature_mask,HEAD_VERSION

def verify(source, migrated, data):
    read=lambda p:json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
    old,new,meta=read(source),read(migrated),read(data/'report.json')
    assert old['feature_version'] in ('combat_features_v5','combat_features_v6') and new['feature_version']=='combat_features_v6'
    weapon_head=new.get('weapon_head')
    if weapon_head:assert weapon_head==HEAD_VERSION and not old.get('weapon_head')
    assert sha(migrated)==meta['model_sha256'] and sha(data/'rollout.jsonl')==meta['rollout_sha256']
    assert sha(data/'sequence.jsonl')==meta['sequence_sha256']
    assert torch.cuda.is_available(), 'CUDA required'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    rows=[json.loads(s) for s in (data/'rollout.jsonl').read_text().splitlines()]
    context=[json.loads(s) for s in (data/'sequence.jsonl').read_text().splitlines()]
    assert len(rows)==meta['rows'] and len(context)==meta['sequence_rows']
    assert all(len(r['features'])==845 and r['sample']['version']==meta['policy_version'] for r in rows)
    prepared=prepare_context(context,rows,'cuda')
    prefix=(prepared[0][:,:,:len(old['actor'][0]['weight'][0])],*prepared[1:])
    report={'device':'cuda','gpu':torch.cuda.get_device_name(),'rows':len(rows),'optimization_steps':0}
    outputs={}
    with torch.no_grad():
        for role in ('actor','value'):
            values=[]
            for model,p,check in ((old,prefix,False),(new,prepared,True)):
                spec=model['attention']
                net=CausalAttention(model[role],spec[role],spec['heads'],spec['window']).cuda()
                y,error=sequence(net,p,32,check,role)
                values.append(y)
                if check:report[role+'_state_max_error']=error
            report[role+'_migration_max_error']=float((values[0]-values[1][:,:values[0].shape[1]]).abs().max())
            # FP32 GEMM reduction changes when the input width changes, even
            # with zero new columns. Use the existing Go/value replay bound;
            # the separate Go FP64 migration test requires exact equality.
            assert report[role+'_migration_max_error']<1e-4, (role, report[role+'_migration_max_error'])
            outputs[role]=values[1]
        z=torch.tensor([r['sample']['latent'] for r in rows],device='cuda')
        attack=torch.tensor([float(r['sample']['attack']) for r in rows],device='cuda')
        vertical=torch.tensor([r['sample']['vertical'] for r in rows],dtype=torch.long,device='cuda')
        std=torch.tensor(new['log_std'],device='cuda')
        weapon=torch.tensor([r['sample'].get('weapon',0) for r in rows],dtype=torch.long,device='cuda') if weapon_head else None
        features=torch.tensor([r['features'] for r in rows],device='cuda')
        mask=feature_mask(features) if weapon_head else None
        lp,_=probabilities(outputs['actor'],std,z,attack,vertical,weapon,mask)
        if weapon_head:
            report['weapon_samples']={str(k):sum(r['sample'].get('weapon',0)==k for r in rows) for k in range(12)}
        expected=torch.tensor([r['sample']['log_probability'] for r in rows],device='cuda')
        report['go_log_probability_max_error']=float((lp-expected).abs().max())
        expected=torch.tensor([r['sample']['value'] for r in rows],device='cuda')
        report['go_value_max_error']=float((outputs['value'][:,0]-expected).abs().max())
        assert report['go_log_probability_max_error']<.003
        assert report['go_value_max_error']<1e-4
        assert max(report['actor_state_max_error'],report['value_state_max_error'])<2e-5
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for key in ('source','model','data','out'):ap.add_argument('--'+key,required=True,type=pathlib.Path)
    args=ap.parse_args()
    result=verify(args.source,args.model,args.data)
    with args.out.open('x',encoding='utf-8') as output:json.dump(result,output,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
