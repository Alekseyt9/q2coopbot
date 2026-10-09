"""Check trained spatial checkpoints against JSON weights on CUDA only."""
import argparse,json,pathlib
from prepare_combat_target_heads import build,forward
from process_combat_architecture_pool import read,save,sha
from train_combat_bc import torch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--roots',nargs='+',type=pathlib.Path,required=True);a=ap.parse_args()
    assert torch.cuda.is_available();torch.manual_seed(20261010)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    for root in a.roots:
        spec=read(root/'weights.json');actor=build(spec,'actor');restored=build(spec,'actor')
        checkpoint=torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=True)
        assert checkpoint['version']=='combat_spatial_aim_bc_checkpoint_v1'
        assert checkpoint['weights_sha256']==sha(root/'weights.json')
        restored.load_state_dict(checkpoint['actor'],strict=True)
        for key,value in actor.state_dict().items():
            assert value.device.type=='cuda' and torch.isfinite(value).all()
            assert torch.equal(value,restored.state_dict()[key]),key
        x=torch.randn(2,8,854,device='cuda')*.05
        with torch.no_grad():
            expected=forward(actor,x);actual=forward(restored,x)
            assert torch.equal(expected,actual) and torch.isfinite(actual).all()
        save(root/'checkpoint-cuda-audit.json',dict(device='cuda',gpu=torch.cuda.get_device_name(),weights_sha256=sha(root/'weights.json'),checkpoint_sha256=sha(root/'checkpoint.pt'),state_exact=True,output_exact=True,scope='CUDA serialized actor restoration only; no Go/CPU numerical comparison.'))
        print(json.dumps(dict(root=str(root),state_exact=True,output_exact=True)),flush=True)

if __name__=='__main__':main()
