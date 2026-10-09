"""CUDA-only restoration of real spatial PPO actor/value/std and Adam state."""
import argparse,json,pathlib
from prepare_combat_target_heads import build,forward
from process_combat_architecture_pool import read,save,sha
from train_combat_bc import torch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--roots',nargs='+',type=pathlib.Path,required=True);a=ap.parse_args();assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    for root in a.roots:
        spec=read(root/'weights.json');report=read(root/'report.json');seal=read(root/'complete.json')
        for filename,field in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:assert sha(root/filename)==seal[field]
        cp=torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=True)
        assert cp['version']=='combat_architecture_checkpoint_v1' and cp['weights_sha256']==sha(root/'weights.json') and report['device']=='cuda'
        modules={name:build(spec,name) for name in ('actor','value')}
        x=[]
        with (root.parent/'rollout/sequence.jsonl').open(encoding='utf-8-sig') as stream:
            for line in stream:
                row=json.loads(line);x.append(row['features'])
                if len(x)==16:break
        assert len(x)==16;x=torch.tensor(x,device='cuda',dtype=torch.float32)[None,:,:]
        for name,module in modules.items():
            restored=build(spec,name);restored.load_state_dict(cp[name],strict=True)
            for key,value in module.state_dict().items():
                assert value.device.type=='cuda' and torch.isfinite(value).all() and torch.equal(value,restored.state_dict()[key]),key
            with torch.no_grad():
                before=forward(module,x);after=forward(restored,x)
                assert torch.isfinite(before).all() and torch.equal(before,after)
        std=torch.nn.Parameter(torch.tensor(spec['log_std'],device='cuda'));assert torch.equal(std.detach(),cp['log_std'])
        for name,parameters in [('actor',list(modules['actor'].parameters())+[std]),('value',list(modules['value'].parameters()))]:
            optimizer=torch.optim.Adam(parameters,lr=cp['config'][name+'_lr']);optimizer.load_state_dict(cp[name+'_optimizer'])
            restored=optimizer.state_dict();saved=cp[name+'_optimizer'];assert restored['param_groups']==saved['param_groups'] and restored['state'].keys()==saved['state'].keys()
            for key,state in saved['state'].items():
                assert state.keys()==restored['state'][key].keys()
                for field,value in state.items():
                    if isinstance(value,torch.Tensor):
                        actual=restored['state'][key][field].to('cuda');assert torch.isfinite(actual).all() and torch.equal(actual,value),field
                    else:assert restored['state'][key][field]==value
        receipt=dict(device='cuda',gpu=torch.cuda.get_device_name(),weights_sha256=sha(root/'weights.json'),checkpoint_sha256=sha(root/'checkpoint.pt'),actor_value_std_exact=True,optimizer_state_exact=True,updates_completed=cp['updates_completed'],consumed_rollouts=cp['consumed_rollouts'],scope='CUDA restored actor/value/std and Adam state; raw outputs exact on16 real sequence feature rows. No CPU/Go numerical model tests. No fresh resumed PPO update, native quality or RNG continuation claim.')
        save(root/'checkpoint-cuda-audit.json',receipt);print(json.dumps(dict(root=str(root),state='passed',updates=cp['updates_completed'])),flush=True)

if __name__=='__main__':main()
