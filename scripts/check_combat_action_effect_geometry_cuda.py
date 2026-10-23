"""Pure Go motor inputs/projection versus CUDA; learned forecasts only on GPU."""
import argparse
import itertools
import json
import subprocess
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('predictor','labels','exporter','out'):ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();a.out.mkdir(parents=True);pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or h==expected,str(path)
        pins[str(path)]=h;return path
    seal=read(pin(a.predictor/'complete.json'));report=read(pin(a.predictor/'report.json',seal['report_sha256']))
    p=read(pin(a.predictor/'predictor.json',seal['predictor_sha256']));assert p['geometry_version']=='combat_predicted_action_geometry_v1'
    assert report['device']=='cuda' and not report['runtime_integrated']
    ls=read(pin(a.labels/'complete.json'));lr=read(pin(a.labels/'report.json',ls['report_sha256']))
    with pin(a.labels/'pure-go-causal.jsonl',lr['pure_go_causal_sha256']).open(encoding='utf-8') as f:
        rows=[json.loads(l) for l in f]
    with pin(a.labels/'requests.jsonl',lr['request_sha256']).open(encoding='utf-8') as f:
        timing=[[q['observation']['observation_age_ms']/300,q['observation']['previous_applied_command']['Msec']/100] for q in map(json.loads,f)]
    from ppo_combat import torch,training_devices
    from combat_action_effect import inputs,FactorizedMotor,projected_geometry
    from process_combat_architecture_pool import compact_closed_exports
    training_devices();torch.set_default_device('cuda');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for name in ('combat_action_effect.py','combat_action_geometry.py'):pin(Path(__file__).with_name(name))
    real=torch.tensor([r['causal'] for r in rows],dtype=torch.float64,device='cuda')
    t=torch.tensor(timing,dtype=torch.float64,device='cuda');assert t.shape==(len(real),2)
    values=[real];times=[t];count=0
    for fine in (False,True):
        pool=torch.nonzero((real[:,151]==0)&(real[:,147]>0)&((real[:,161]>0)==fine)).flatten()
        if not len(pool):continue
        ids=pool[torch.linspace(0,len(pool)-1,steps=min(64,len(pool)),device='cuda').round().long()]
        selected=real[ids];limit=15. if fine else 180.
        for i,(yaw,pitch) in enumerate(itertools.product([0,-5,-2.5,2.5,5],repeat=2)):
            c=selected.clone()
            if i:
                for axis,offset in enumerate((yaw,pitch)):c[:,164+axis]=((c[:,164+axis]*180+offset)/limit).clamp(-1+1e-6,1-1e-6)*limit/180
            values.append(c);times.append(t[ids]);count+=len(c)
    c=torch.cat(values);t=torch.cat(times);m=p['motor']
    def decode(dtype):
        net=FactorizedMotor().to(dtype=dtype)
        with torch.no_grad():
            for branch,key in ((net.position,'position_layers'),(net.direction,'direction_layers')):
                for i,l in zip((0,2),m[key]):
                    branch[i].weight.copy_(torch.tensor(l['weight'],dtype=dtype,device='cuda'));branch[i].bias.copy_(torch.tensor(l['bias'],dtype=dtype,device='cuda'))
        return net
    with torch.no_grad():
        net64,net32=decode(torch.float64),decode(torch.float32)
        expected_inputs=inputs(c,t)
        forecast=net64((expected_inputs-torch.tensor(m['mean'],dtype=torch.float64,device='cuda'))/torch.tensor(m['std'],dtype=torch.float64,device='cuda'))
        f32=net32((inputs(c.float(),t.float())-torch.tensor(m['mean'],device='cuda'))/torch.tensor(m['std'],device='cuda'))
        expected64=projected_geometry(c,forecast,False)
        expected32=projected_geometry(c.float(),f32,False).double()
    ip,op=a.out/'inputs.jsonl',a.out/'go-projection.jsonl'
    with ip.open('w',encoding='utf-8') as f:
        for causal,time,delta in zip(c.tolist(),t.tolist(),forecast.tolist()):
            f.write(json.dumps(dict(causal=causal,timing=time,forecast=delta),separators=(',',':'))+'\n')
    pin(a.exporter)
    subprocess.run([str(a.exporter.resolve()),'--input',str(ip.resolve()),'--out',str(op.resolve()),
        '--geometry-version',p['geometry_version'],'--motor-inputs'],check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    with op.open(encoding='utf-8') as f:rows=[json.loads(line) for line in f]
    actual=torch.tensor([r['geometry'] for r in rows],dtype=torch.float64,device='cuda')
    actual_inputs=torch.tensor([r['motor_inputs'] for r in rows],dtype=torch.float64,device='cuda')
    input_error=float((actual_inputs-expected_inputs).abs().max());error64=float((actual-expected64).abs().max())
    continuous=[i for i in range(29) if i not in (15,28)]
    error32=float((actual-expected32).abs().max());cont=float((actual[:,continuous]-expected32[:,continuous]).abs().max())
    soft=float((actual[:,[15,28]]-expected32[:,[15,28]]).abs().max());motor_error=float((forecast-f32.double()).abs().max())
    assert input_error<1e-9 and error64<1e-7,(input_error,error64)
    # +/-1unit soft band:0.005 feature units correspond to0.01world units,
    # below protocol position quantization of0.125. Continuous tolerance unchanged.
    assert cont<.0002 and soft<.005 and motor_error<.00001,(cont,soft,motor_error)
    compact_closed_exports(a.out)
    for path,h in pins.items():assert sha(Path(path))==h,path
    report=dict(state='complete',version='combat_action_effect_geometry_parity_v1',device='cuda',geometry_version=p['geometry_version'],
        real_inputs=len(real),candidate_inputs=count,motor_input_max_error=input_error,max_error_fp64=error64,max_error_fp32=error32,
        max_error_fp32_continuous=cont,max_error_fp32_soft=soft,max_motor_error_fp32=motor_error,
        inputs=pins,script_sha256=sha(__file__),go_neural_execution=False,training_performed=False,promotion=False,
        scope='Go computes only observed motor inputs and pure projection of CUDA-produced forecasts. Every learned forecast executed on CUDA. Does not verify Go NN scoring; separate actual-game live CUDA audit required.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','scope')}))


if __name__=='__main__':main()
