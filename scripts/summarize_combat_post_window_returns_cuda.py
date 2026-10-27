"""Observed finite rewards on CUDA; not a bootstrapped Q or training target."""
import argparse
import json
from pathlib import Path

from ppo_combat import torch, training_devices
from process_combat_architecture_pool import read, save, sha


def run(future, original, out):
    assert not out.exists()
    training_devices()
    pins={str(Path(__file__).resolve()):sha(Path(__file__))}
    seals=[(future/'complete.json',future/'report.json'),(original/'long-complete.json',original/'long-report.json')]
    for seal_path,report_path in seals:
        seal=read(seal_path)
        assert seal['state']=='complete' and sha(report_path)==seal['report_sha256']
        pins[str(seal_path)]=sha(seal_path);pins[str(report_path)]=sha(report_path)
    def utility(folder):
        report=folder/'report.json';meta=read(report)
        proof=folder/'cuda-verification.json'
        assert read(proof)['state']=='passed' and read(proof)['device']=='cuda'
        assert meta['reward_version']=='combat_reward_v4'
        path=folder/'rollout.jsonl'
        assert sha(path)==meta['rollout_sha256']
        for p in (report,proof,path):pins[str(p)]=sha(p)
        rows=[r for r in map(json.loads,path.open(encoding='utf-8-sig')) if r['frame']>=374]
        frames=[r['frame'] for r in rows]
        assert frames and frames==list(range(374,374+len(frames)))
        rewards=torch.tensor([r['reward'] for r in rows],device='cuda',dtype=torch.float64)
        discounts=torch.pow(torch.tensor(.99,device='cuda',dtype=torch.float64),torch.arange(len(rows),device='cuda',dtype=torch.float64))
        return dict(value=float((rewards*discounts).sum()),undiscounted=float(rewards.sum()),
            frames=len(rows),reward_sha256=meta['reward_config_sha256'])
    cases=[]
    expected_reward=None
    for index in range(3):
        a,b=utility(future/f'case-{index}/cuda'),utility(future/f'case-{index+3}/cuda')
        assert a['reward_sha256']==b['reward_sha256']
        expected_reward=a['reward_sha256']
        cases.append(dict(stream=f'fork{index+1}',control=a,back=b,delta_back_minus_control=b['value']-a['value']))
    a,b=utility(original/'case-0/cuda'),utility(original/'case-6/cuda')
    assert a['reward_sha256']==b['reward_sha256']==expected_reward
    cases.insert(0,dict(stream='original',control=a,back=b,delta_back_minus_control=b['value']-a['value']))
    delta=torch.tensor([r['delta_back_minus_control'] for r in cases],device='cuda',dtype=torch.float64)
    summary=dict(mean_delta=float(delta.mean()),sample_stddev=float(delta.std(correction=1)),
        positive_streams=int((delta>0).sum()),negative_streams=int((delta<0).sum()),
        new_suffix_mean_delta=float(delta[1:].mean()),streams=4,worlds=1)
    assert all(sha(Path(p))==digest for p,digest in pins.items())
    out.mkdir(parents=True)
    save(out/'report.json',dict(state='complete',device='cuda',gamma=.99,from_frame=374,cases=cases,summary=summary,inputs=pins,
        scope='Observed finite reward_v4 only, no critic bootstrap or unobserved future. One selected validation world and correlated option prefixes; four stochastic streams, not independent worlds or calibrated Q. No hard teacher target, actor fitting or promotion.'))
    save(out/'complete.json',dict(state='complete',report_sha256=sha(out/'report.json')))
    print(json.dumps(dict(cases=cases,summary=summary),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('future','original','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    run(a.future.resolve(),a.original.resolve(),a.out.resolve())
