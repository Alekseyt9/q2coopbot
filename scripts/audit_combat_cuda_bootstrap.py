"""Prototype CUDA bootstrap against existing recorded values, no Go execution."""
import argparse,json,pathlib,time
from prepare_combat_target_heads import build
from process_combat_architecture_pool import read,save,sha
from ppo_recurrent import prepare_context
from combat_cuda_bootstrap import next_values
from train_combat_bc import torch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=pathlib.Path,required=True);ap.add_argument('--data',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
    assert torch.cuda.is_available() and not a.out.exists();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    meta=read(a.data/'report.json');assert meta['model_sha256']==sha(a.model)
    assert sha(a.data/'rollout.jsonl')==meta['rollout_sha256'] and sha(a.data/'sequence.jsonl')==meta['sequence_sha256']
    rows=[json.loads(s) for s in (a.data/'rollout.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    context=[json.loads(s) for s in (a.data/'sequence.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    by_frame={(c['seed'],c['frame']):c for c in context};assert len(by_frame)==len(context)
    selected=[i for i,r in enumerate(rows) if not r['terminal'] and not r['truncated'] and r['next_frame']==r['frame']+1 and (r['seed'],r['next_frame']) in by_frame]
    assert selected
    value=build(read(a.model),'value');prepared=prepare_context(context,rows,'cuda')
    features=torch.tensor([by_frame[(rows[i]['seed'],rows[i]['next_frame'])]['features'] for i in selected],device='cuda',dtype=torch.float32)
    chosen=torch.tensor(selected,device='cuda',dtype=torch.long);reference=torch.tensor([rows[i]['next_value'] for i in selected],device='cuda',dtype=torch.float32)
    with torch.no_grad():
        start=time.perf_counter();actual=next_values(value,features,prepared,chosen);torch.cuda.synchronize();seconds=time.perf_counter()-start
        error=(actual-reference).abs();assert torch.isfinite(actual).all()
    report=dict(state='passed' if float(error.max())<1e-4 else 'mismatch',device='cuda',rows=len(rows),checked_rows=len(selected),excluded_rows=len(rows)-len(selected),max_error=float(error.max()),mean_error=float(error.mean()),seconds=seconds,model_sha256=sha(a.model),rollout_sha256=meta['rollout_sha256'],sequence_sha256=meta['sequence_sha256'],helper_sha256=sha(pathlib.Path(__file__).with_name('combat_cuda_bootstrap.py')),scope='Prototype only on adjacent nonterminal/nontruncated provider contexts. Excluded boundary/reset/handoff rows remain unverified. Existing recorded next_value references, no Go model execution. Next features provisionally taken from adjacent context; production native-only exporter still needed to emit exact Step.Next features and reset metadata. No training pipeline replacement yet.')
    save(a.out,report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
