"""Native movement/pose support audit; metadata and features only, no neural work."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from audit_combat_contact_loss_probe import load_steps
from evaluate_registered_combat_models import read,save,sha


def route_active(x):
    return (len(x)==1554 and x[0]>0 and x[1489]==1 and not any(x[73+12*j] for j in range(8))
            and any(x[881+30*j]==1 and x[1490+8*j]==1 and x[1491+8*j]==1 for j in range(8)))


def increment(count,features,step,selected=None):
    action,command=step['applied_action'],step['sent_command']
    assert step['native_step'] and step['server_execution']['matched'] and step['server_execution']['window_exclusive']
    assert action['identity']==step['observation']['identity']
    assert abs(action['forward']-command['Forward']/400)<1e-6 and abs(action['side']-command['Side']/400)<1e-6
    assert action['vertical'] in ('release','jump','crouch')
    assert command['Up']=={'release':0,'jump':400,'crouch':-400}[action['vertical']]
    context='route' if route_active(features) else ('visible' if any(features[73+12*j] for j in range(8)) else 'no_visible_no_route')
    count['rows']+=1;count[context+'_rows']+=1
    count['applied_'+action['vertical']]+=1;count[context+'_applied_'+action['vertical']]+=1
    count[context+'_native_ducked']+=bool(step['observation'].get('ducked'))
    if selected is not None:
        pose=('release','jump','crouch')[selected]
        count['selected_'+pose]+=1;count[context+'_selected_'+pose]+=1


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--demonstrations',required=True,type=Path)
    ap.add_argument('--screens',required=True,type=Path,nargs='+')
    ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p):p=Path(p).resolve();pins[str(p)]=sha(p);return p
    teacher=read(pin(a.demonstrations));assert teacher['state']=='passed' and teacher['ppo_eligible'] is False
    assert teacher['version']=='combat_movement_demonstrations_v1'
    for p,h in teacher['inputs'].items():assert sha(p)==h
    demonstrations={s:Counter() for s in ('train','diagnostic')};episodes=[]
    for e in teacher['episodes']:
        pack=pin(e['pack']);assert sha(pack)==e['pack_sha256']
        data=np.load(pack,allow_pickle=False)
        observations=load_steps(pin(pack.parent/'observations.jsonl'))
        steps=load_steps(pin(pack.parent/'native/steps.jsonl'))
        by_frame={s['observation']['identity']['frame']:s for s in steps}
        assert len(observations)==e['context_rows']==len(data['features'])
        assert len(data['indices'])==e['rows']
        c=Counter()
        for j,index in enumerate(data['indices']):
            o=observations[index];step=by_frame[o['identity']['frame']]
            assert step['observation']==o and step['owner']=='rules'
            assert np.allclose(data['movement'][j],[step['applied_action']['forward'],step['applied_action']['side']],atol=1e-7,rtol=0)
            increment(c,data['features'][index],step)
        demonstrations[e['split']].update(c)
        episodes.append(dict(map=e['map'],seed=e['seed'],split=e['split'],counts=dict(c)))
    screens=[]
    for root in a.screens:
        root=root.resolve();spec=read(pin(root/'spec.json'));quality=read(pin(root/'acceptance.json'))
        assert quality['state']=='passed'
        for p,h in quality['inputs'].items():assert sha(p)==h
        cases=[];totals={}
        for v in spec['variants']:
            if v['label']=='rules':continue
            d=root/f'case-{v["index"]:02d}'
            meta=read(pin(d/'cuda/report.json'));proof=read(pin(d/'cuda/cuda-verification.json'))
            assert meta['model_sha256']==v['model_sha256']==sha(v['model']) and proof['state']=='passed'
            rows=load_steps(pin(d/'cuda/rollout.jsonl'))
            assert sha(d/'cuda/rollout.jsonl')==meta['rollout_sha256']
            path=next((d/'native').glob('replay-*/steps.jsonl'));steps=load_steps(pin(path))
            native={s['observation']['identity']['frame']:s for s in steps if s['owner']=='provider'}
            c=Counter()
            for r in rows:
                s=native[r['frame']]
                assert r['seed']==v['context']['seed']
                increment(c,r['features'],s,r['sample']['vertical'])
            totals.setdefault(v['label'],Counter()).update(c)
            cases.append(dict(map=v['context']['map'],seed=v['context']['seed'],role=v['label'],counts=dict(c)))
        screens.append(dict(root=str(root),counts={k:dict(v) for k,v in totals.items()},cases=cases))
    assert all(sha(p)==h for p,h in pins.items())
    train=demonstrations['train']
    save(a.out,dict(version='combat_movement_native_support_audit_v1',state='passed',neural_execution=False,
        demonstrations={k:dict(v) for k,v in demonstrations.items()},episodes=episodes,screens=screens,inputs=pins,
        route_demonstration_support=bool(train['route_rows']),
        demonstrated_pose_classes=[p for p in ('release','jump','crouch') if train['applied_'+p]],
        scope='Support and actual executed pose counts, not optimal-action labels or causal proof of losses. Zero route rows means gated route BC had no active examples. No quality promotion.'))
    print({k:dict(v) for k,v in demonstrations.items()},flush=True)


if __name__=='__main__':main()
