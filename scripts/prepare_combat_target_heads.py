"""Migrate architecture variants and audit the new action contract on CUDA."""
import argparse,pathlib,subprocess
from process_combat_architecture_pool import read,sha,save
from ppo_recurrent import torch,network,Recurrent,CausalAttention
from combat_target_head import migrate,probabilities,distribution,selected_means

def build(model,name):
    if model.get('memory'): module=Recurrent(model[name],model['memory'][name])
    elif model.get('attention'):
        spec=model['attention'];module=CausalAttention(model[name],spec[name],spec['heads'],spec['window'])
    else:module=network(model[name])
    if name=='actor' and model.get('spatial_aim'):
        from combat_spatial_aim import wrap
        module=wrap(module,model)
    return module.to('cuda')

def forward(module,x):
    output=module(x)
    return output[0] if isinstance(output,tuple) else output

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--compiler',type=pathlib.Path,required=True)
    a=ap.parse_args();assert torch.cuda.is_available() and not a.out.exists()
    torch.manual_seed(20261009);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    entries=[e for e in read(a.protocol)['evaluations'] if e['label']=='after' and e['model']!='firebc']
    assert len(entries)==8
    a.out.mkdir();reports=[];plans=[]
    for entry in entries:
        sourceplan=read(entry['plan']);path=pathlib.Path(sourceplan['model_path'])
        assert sha(path)==entry['deterministic_weights_sha256']
        old=read(path);new=migrate(old)
        x=torch.randn(3,7,845,device='cuda')*.1
        intent=torch.nn.functional.one_hot(torch.zeros(3,7,dtype=torch.long,device='cuda'),9).float()
        extended=torch.cat((x,intent),-1)
        errors={}
        with torch.no_grad():
            for name in ('actor','value'):
                before=forward(build(old,name),x);after=forward(build(new,name),extended)
                error=float((before-after[...,:before.shape[-1]]).abs().max())
                assert error<3e-5,(entry['model'],name,error);errors[name]=error
                if name=='actor':
                    for slot in range(8):assert float((after[...,29+2*slot:31+2*slot]-before[...,2:4]).abs().max())<3e-5
        raw=torch.randn(4,45,device='cuda',requires_grad=True);features=torch.zeros(4,854,device='cuda')
        features[:,833]=1;features[1:,426]=1;features[2:,431]=1
        target=torch.tensor([0,1,2,1],dtype=torch.long,device='cuda')
        std=torch.tensor(new['log_std'],device='cuda');z=selected_means(raw,target).detach()+.1
        lp,ent=probabilities(raw,std,z,torch.zeros(4,device='cuda'),torch.zeros(4,dtype=torch.long,device='cuda'),torch.zeros(4,dtype=torch.long,device='cuda'),features,target)
        assert bool(torch.isfinite(lp).all() & torch.isfinite(ent).all())
        (-lp.mean()).backward();assert bool(torch.isfinite(raw.grad).all())
        assert bool((raw.grad[:,33:45]==0).all()) and bool((raw.grad[:,23:29]==0).all())
        assert distribution(raw,features).probs[0,0]==1
        try: probabilities(raw,std,z,torch.zeros(4,device='cuda'),torch.zeros(4,dtype=torch.long,device='cuda'),torch.zeros(4,dtype=torch.long,device='cuda'),features,torch.full((4,),8,dtype=torch.long,device='cuda'))
        except ValueError:pass
        else:raise AssertionError('Unavailable target accepted')
        folder=a.out/entry['model'];folder.mkdir();weights=folder/'weights.json';save(weights,new)
        # Bind current registry/runner metadata rather than copying stale plans.
        chosen=sourceplan['tasks'][:2];planpath=folder/'plan.json'
        split=chosen[0]['split'];start=chosen[0]['episode']['splits'][split]['start']
        subprocess.run([str(a.compiler.resolve()),'--registry',sourceplan['registry_path'],
            '--episodes',','.join(t['episode']['id'] for t in chosen),'--split',split,
            '--mode','learned','--model',str(weights.resolve()),'--count','4',
            '--seed-offset',str(chosen[0]['seeds'][0]-start),
            '--root',str(pathlib.Path(__file__).resolve().parents[1]),'--out',str(planpath.resolve()),
            '--artifacts',str((folder/'capture').resolve())],check=True,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),stdout=subprocess.DEVNULL)
        plans.append(str(planpath.resolve()))
        reports.append(dict(model=entry['model'],architecture='gru' if old.get('memory') else 'attention' if old.get('attention') else 'mlp',parent_sha256=sha(path),weights_sha256=sha(weights),maximum_error=errors,masked_gradients=True,unavailable_sample_rejected=True,optimizer_reset_required=True))
    save(a.out/'plans.json',plans);save(a.out/'cuda-audit.json',dict(state='passed',device='cuda',gpu=torch.cuda.get_device_name(),models=reports,scope='Migration and action likelihood checks; no new training or quality claim'))
    print('CUDA target-head audit passed for',len(reports),'models',flush=True)

if __name__=='__main__':main()
