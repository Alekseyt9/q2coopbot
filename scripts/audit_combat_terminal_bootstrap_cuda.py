"""Compare masked CUDA bootstrap with the full-batch critic on a real export."""
import argparse,copy,json
from pathlib import Path
from process_combat_architecture_pool import read,save,sha
from prepare_combat_target_heads import build
from ppo_recurrent import prepare_context,torch
from combat_cuda_bootstrap import bootstrap_rows,next_values

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('model','data','out'):ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args();assert torch.cuda.is_available() and not a.out.exists()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    meta=read(a.data/'report.json');assert meta['model_sha256']==sha(a.model)
    rollout=a.data/('native-rollout.jsonl' if (a.data/'native-rollout.jsonl').exists() else 'rollout.jsonl')
    rows=[json.loads(line) for line in rollout.open(encoding='utf-8-sig')]
    context=[json.loads(line) for line in (a.data/'sequence.jsonl').open(encoding='utf-8-sig')]
    assert any(r['terminal'] and r['bootstrap_zero'] and 'next_features' not in r for r in rows)
    errors={}
    for dtype in (torch.float32,torch.float64):
        prepared=prepare_context(context,rows,'cuda',dtype=dtype)
        value=build(read(a.model),'value').to(dtype=dtype)
        for pattern in ('native','mixed','all_zero'):
            selected=copy.deepcopy(rows)
            for i,r in enumerate(selected):
                if (pattern=='mixed' and i%2==0) or pattern=='all_zero':
                    r['bootstrap_zero']=True;r['terminal']=True;r.pop('next_features',None)
            # Historical full-batch computation used features for all rows,
            # then zeroed terminal values. Valid stand-ins below are reference
            # inputs only; masked rows must not affect any live prediction.
            nx=torch.tensor([r.get('next_features',r['features']) for r in selected],device='cuda',dtype=dtype)
            reset=torch.tensor([r['next_reset'] for r in selected],device='cuda',dtype=torch.bool)
            zero=torch.tensor([r['bootstrap_zero'] for r in selected],device='cuda',dtype=torch.bool)
            with torch.no_grad():
                expected=next_values(value,nx,prepared,reset=reset);expected[zero]=0
                actual=bootstrap_rows(value,selected,prepared)
            error=float((actual-expected).abs().max());assert error<1e-4,(dtype,pattern,error)
            assert bool((actual[zero]==0).all())
            errors[str(dtype)+'_'+pattern]=error
        # All-terminal rows have deliberately invalid/unavailable features.
        # A critic that must never run catches even a discarded evaluation.
        class Forbidden(torch.nn.Module):
            def __init__(self):super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1,device='cuda',dtype=dtype))
            def forward(self,*args,**kwargs):raise AssertionError('Terminal critic executed')
        poison=[dict(terminal=True,bootstrap_zero=True,next_features=[float('nan')])]
        assert float(bootstrap_rows(Forbidden(),poison)[0])==0
    save(a.out,dict(state='passed',device='cuda',rows=len(rows),errors=errors,
        model_sha256=sha(a.model),native_export_report_sha256=sha(a.data/'report.json'),
        helper_sha256=sha(Path(__file__).with_name('combat_cuda_bootstrap.py')),
        scope='Real causal critic: native/mixed/all-zero masks versus full-batch reference, float32/float64. Missing terminal features and forbidden critic checked. No fitting or Go neural inference.'))
    print(json.dumps(errors),flush=True)

if __name__=='__main__':main()
