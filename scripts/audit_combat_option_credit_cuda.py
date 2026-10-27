"""CUDA credit/exploration diagnostic on verified option suffixes; no fitting."""
import argparse
import json
from pathlib import Path

from ppo_combat import torch, training_devices
from process_combat_architecture_pool import read, save, sha


def run(root,returns,out):
    assert not out.exists()
    training_devices()
    seal=read(returns/'complete.json');summary=read(returns/'report.json')
    assert sha(returns/'report.json')==seal['report_sha256'] and summary['state']=='complete'
    proof=read(root/'complete.json');assert sha(root/'report.json')==proof['report_sha256']
    pins={str(root/'report.json'):sha(root/'report.json'),str(returns/'report.json'):sha(returns/'report.json'),str(Path(__file__).resolve()):sha(Path(__file__))}
    def load(index):
        folder=root/f'case-{index}/cuda';meta=read(folder/'report.json');path=folder/'rollout.jsonl'
        assert read(folder/'cuda-verification.json')['state']=='passed' and sha(path)==meta['rollout_sha256']
        pins[str(path)]=sha(path)
        rows=[r for r in map(json.loads,path.open(encoding='utf-8-sig')) if r['frame']>=374]
        assert [r['frame'] for r in rows]==list(range(374,374+len(rows)))
        return rows
    def gae(rows,lam):
        reward=torch.tensor([r['reward'] for r in rows],device='cuda',dtype=torch.float64)
        values=torch.tensor([r['sample']['value'] for r in rows],device='cuda',dtype=torch.float64)
        nxt=torch.tensor([r['next_value'] for r in rows],device='cuda',dtype=torch.float64)
        terminal=torch.tensor([r['terminal'] for r in rows],device='cuda',dtype=torch.bool)
        delta=reward+.99*torch.where(terminal,0.,nxt)-values
        advantage=torch.zeros_like(delta)
        for i in range(len(rows)-1,-1,-1):
            continuation=i+1<len(rows) and rows[i+1]['index']==rows[i]['index']+1 and not rows[i]['terminal'] and not rows[i]['truncated']
            advantage[i]=delta[i]+(.99*lam*advantage[i+1] if continuation else 0.)
        return float(advantage[0])
    records=[]
    for index in range(3):
        a,b=load(index),load(index+3)
        assert a[0]['sample']['value']==b[0]['sample']['value'], 'Different initial state critic'
        records.append(dict(stream=index+1,gae095_delta_back_minus_control=gae(b,.95)-gae(a,.95),
            gae1_delta_back_minus_control=gae(b,1.)-gae(a,1.),root_value=a[0]['sample']['value']))
    back=load(3)
    model_path=next(Path(p) for p in read(root/'case-3/native/report.json')['source_sha256'] if p.endswith('.json') and 'back-' in p)
    model=read(model_path);pins[str(model_path)]=sha(model_path)
    std=torch.tensor(model['log_std'][:2],device='cuda',dtype=torch.float64).exp()
    intervention=[r['sample']['movement_probe'] for r in back if 374<=r['frame']<379]
    assert len(intervention)==5
    shifts=torch.tensor([r['mean_shift'] for r in intervention],device='cuda',dtype=torch.float64)
    noise=torch.tensor([r['gaussian_noise'][:2] for r in intervention],device='cuda',dtype=torch.float64)
    diagnostics=dict(movement_std=std.cpu().tolist(),shift_over_std=(shifts[0]/std).cpu().tolist(),
        five_frame_movement_kl=float((.5*(shifts/std).square()).sum()),
        observed_five_frame_movement_log_density_ratio_shifted_over_base=float((((noise+shifts).square()-noise.square())/(2*std.square())).sum()),
        gae095_weight_at50frames=.9405**50,gae095_weight_at100frames=.9405**100)
    assert all(sha(Path(p))==digest for p,digest in pins.items())
    out.mkdir(parents=True)
    save(out/'report.json',dict(state='complete',device='cuda',records=records,diagnostics=diagnostics,inputs=pins,
        scope='Existing diagnostic suffixes only. GAE mirrors gamma.99 and terminal/truncation semantics, includes stored critic bootstrap. Lambda1 is a diagnostic, not a fit recommendation. Movement marginal KL/density on recorded five-frame intervention is not the probability of useful behavior. No actor update or promotion.'))
    print(json.dumps(dict(records=records,diagnostics=diagnostics),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','returns','out'):p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args();run(a.root.resolve(),a.returns.resolve(),a.out.resolve())
