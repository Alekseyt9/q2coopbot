"""Compare pure Go geometry with CUDA; no Go neural inference or scoring."""
import argparse
import itertools
import json
import subprocess
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('data','exporter','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();assert not args.out.exists();args.out.mkdir(parents=True)
    seal=read(args.data/'complete.json');meta=read(args.data/'report.json')
    assert sha(args.data/'report.json')==seal['report_sha256'] and sha(args.data/'dataset.pt')==meta['dataset_sha256']
    pins={str(p.resolve()):sha(p) for p in (args.data/'complete.json',args.data/'report.json',args.data/'dataset.pt',args.exporter,
        Path(__file__).with_name('combat_action_geometry.py'))}
    from ppo_combat import torch,training_devices
    training_devices();torch.set_default_device('cuda')
    from combat_action_geometry import features_for_version,names_for_version,VERSION
    from process_combat_architecture_pool import compact_closed_exports
    data=torch.load(args.data/'dataset.pt',map_location='cuda',weights_only=True)
    version=data['geometry_version'];NAMES=names_for_version(version)
    features=lambda x:features_for_version(x,version)
    base=data['features_with_recoil'].double();parts=[base];count=0
    for fine in (False,True):
        pool=torch.nonzero((data['action_geometry'][:,0]>0)&((base[:,161]>0)==fine)).flatten()
        indices=pool[torch.linspace(0,len(pool)-1,steps=min(64,len(pool)),device='cuda').round().long()]
        selected=base[indices];limit=15/180 if fine else 1.
        for index,(yaw,pitch) in enumerate(itertools.product([0,-5,-2.5,2.5,5],repeat=2)):
            candidate=selected.clone()
            if index:
                for axis,offset in enumerate((yaw,pitch)):
                    ratio=((candidate[:,164+axis]*180+offset)/(limit*180)).clamp(-1+1e-6,1-1e-6)
                    candidate[:,164+axis]=ratio*limit
            parts.append(candidate);count+=len(candidate)
    values=torch.cat(parts);expected64=features(values);expected32=features(values.float()).double()
    input_path=args.out/'inputs.jsonl';output=args.out/'go-geometry.jsonl'
    with input_path.open('w',encoding='utf-8') as f:
        for row in values.tolist():f.write(json.dumps(dict(causal=row),separators=(',',':'))+'\n')
    command=[str(args.exporter.resolve()),'--input',str(input_path.resolve()),'--out',str(output.resolve()),'--geometry-version',version]
    subprocess.run(command,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    with output.open(encoding='utf-8') as f:rows=[json.loads(line) for line in f]
    actual=torch.tensor([r['geometry'] for r in rows],dtype=torch.float64,device='cuda')
    assert torch.equal(torch.tensor([r['causal'] for r in rows],dtype=torch.float64,device='cuda'),values)
    error64=float((actual-expected64).abs().max());error32=float((actual-expected32).abs().max())
    continuous=[i for i in range(29) if i not in (15,28)]
    continuous_error=float((actual[:,continuous]-expected32[:,continuous]).abs().max())
    soft_error=float((actual[:,[15,28]]-expected32[:,[15,28]]).abs().max())
    if version==VERSION:
        discrete=[NAMES.index('eye_ray_bbox_intersection'),NAMES.index('muzzle_ray_bbox_intersection')]
        assert torch.equal(actual[:,discrete],expected64[:,discrete]),'FP64 bbox intersection mismatch'
        assert torch.equal(actual[:,discrete],expected32[:,discrete]),'FP32 bbox intersection mismatch'
    assert error64<1e-7,error64
    assert continuous_error<.0002,continuous_error
    # Soft coordinates map two world units to [0,1]. 0.001 here is 0.002
    # world units, far below observed position quantization (1/8 unit).
    assert soft_error<.001,soft_error
    compact_closed_exports(args.out)
    for path,digest in pins.items():assert sha(Path(path))==digest,path
    report=dict(state='complete',version='combat_action_geometry_parity_v1',device='cuda',real_inputs=len(base),candidate_inputs=count,
        max_error_fp64=error64,max_error_fp32=error32,bbox_intersections_exact=version==VERSION,geometry_version=version,
        max_error_fp32_continuous=continuous_error,max_error_fp32_soft=soft_error,
        inputs=pins,script_sha256=sha(__file__),go_neural_execution=False,training_performed=False,promotion=False,
        scope='Pure observation/action geometry only in Go. CUDA FP64 reference and training FP32 compared. No learned scores, no policy Decisions and no Go neural benchmark.')
    save(args.out/'report.json',report);save(args.out/'complete.json',dict(state='complete',report_sha256=sha(args.out/'report.json')))
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','scope')}))


if __name__=='__main__':main()
