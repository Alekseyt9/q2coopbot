"""Seal honest weights-only evaluation artifacts after native zero compatibility."""
import argparse
import shutil
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('models','compatibility','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists()
    models=read(a.models/'report.json');assert sha(a.models/'report.json')==read(a.models/'complete.json')['report_sha256']
    compatible=read(a.compatibility);assert compatible['gate_passed'] and compatible['native_cases']==compatible['cuda_cases']==8
    assert models['device']=='cuda' and models['source_parameters_exact'] and models['zero_source_raw_exact']
    a.out.mkdir(parents=True)
    for label in ('noise-only','inertia'):
        model=models['models'][label];assert sha(model['path'])==model['sha256'];target=a.out/label;target.mkdir();shutil.copyfile(model['path'],target/'weights.json')
        save(target/'report.json',dict(state='complete',version='combat_experimental_policy_weights_v1',device='cuda',evaluation_only=True,ppo_checkpoint_written=False,
            native_compatibility_report=str(a.compatibility.resolve()),native_compatibility_sha256=sha(a.compatibility),actor_preparation_report=str((a.models/'report.json').resolve()),actor_preparation_sha256=sha(a.models/'report.json'),
            source_weights_sha256=models['source_weights_sha256'],runtime_nonzero_parity_proven=False,training_performed=False,promotion=False,
            scope='Fixed mean/variance architecture intervention with unchanged neural tensors. Native nonzero Go/CUDA replay is required in the actual evaluation, not asserted by zero receipt. No resumable PPO checkpoint.'))
        save(target/'complete.json',dict(state='complete',weights_sha256=sha(target/'weights.json'),report_sha256=sha(target/'report.json')))
    print(a.out)

if __name__=='__main__':main()
