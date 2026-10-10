"""Prepare common coarse/fine spatial residuals, auditing all variants on CUDA."""
import argparse,copy,pathlib
from process_combat_architecture_pool import read,save,sha
from prepare_combat_target_heads import build,forward
from combat_precision_head import migrate,validate_precision_model
from combat_spatial_aim import migrate_coarse,validate,COARSE_VERSION
from train_combat_bc import torch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[1];a.out=a.out.resolve();assert not a.out.exists() and torch.cuda.is_available();a.out.mkdir(parents=True)
    torch.set_num_threads(2);torch.manual_seed(20261010)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    parents=[(f'm{i}',repo/f'workspace/artifacts/target-head-v1-20261009/m{i}/target-bc/weights.json') for i in range(8)]
    parents.append(('parent3',repo/'workspace/artifacts/first-life-update4-capture-20261010/control-weights.json'))
    audit=[]
    for name,path in parents:
        source=read(path);precision=source if source.get('aim_mode_head') else migrate(source)
        validate_precision_model(precision);candidate=migrate_coarse(precision);validate(candidate)
        assert candidate['spatial_aim']['version']==COARSE_VERSION
        assert source['value']==candidate['value'] and source['log_std']==candidate['log_std']
        x=torch.randn(3,8,854,device='cuda')*.1;errors={}
        with torch.no_grad():
            for head in ('actor','value'):
                before=forward(build(precision,head),x);after=forward(build(candidate,head),x)
                errors[head]=float((before-after).abs().max());assert errors[head]<3e-5
        # The new coarse residual must influence only the eight coarse pairs.
        changed=copy.deepcopy(candidate);changed['spatial_aim']['layers'][-1]['bias'][4]=.1
        with torch.no_grad():
            baseline=forward(build(candidate,'actor'),x);delta=forward(build(changed,'actor'),x)-baseline
            expected=torch.zeros_like(delta);expected[...,29:45:2]=.1
            assert float((delta-expected).abs().max())<3e-5
        folder=a.out/name;folder.mkdir();save(folder/'weights.json',candidate)
        audit.append(dict(model=name,architecture='gru' if source.get('memory') else 'attention' if source.get('attention') else 'mlp',parent=str(path),parent_sha256=sha(path),weights=str(folder/'weights.json'),weights_sha256=sha(folder/'weights.json'),maximum_migration_error=errors,coarse_output_mapping=True,inherited_value_and_std_preserved=True))
    save(a.out/'cuda-migration-audit.json',dict(state='passed',device='cuda',gpu=torch.cuda.get_device_name(),models=audit,
        scope='Eight architecture priors plus strongest retained PPO parent. Zero coarse residual migration preserves every existing output; coarse pair mapping tested on CUDA. No training or gameplay quality improvement claim. New BC optimizer required; original PPO checkpoint remains untouched.'))
    print('CUDA migration and coarse mapping passed:',len(audit),'models',flush=True)

if __name__=='__main__':main()
