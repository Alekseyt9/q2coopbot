"""Seal later native MG contexts and fresh seed-group splits before alternatives."""
import argparse
import collections
import json
from pathlib import Path
import re
from evaluate_registered_combat_models import read,save,sha
from report_combat_machinegun_hits import scan
from report_registered_machinegun_outcomes import outcome


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('baseline','parent-dataset','protocol','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    closed=read(pin(a.baseline/'complete.json'));assert closed['native_cases']==closed['cuda_cases']==16
    spec=read(pin(a.baseline/'spec.json',closed['spec_sha256']));source=Path(spec['source']);seal=read(pin(source/'complete.json'))
    pin(source/'weights.json',seal['weights_sha256']);pin(a.protocol)
    parent_seal=read(pin(a.parent_dataset/'complete.json'));parent=read(pin(a.parent_dataset/'report.json',parent_seal['report_sha256']))
    assert parent['contexts']==17 and parent['source_weights_sha256']==seal['weights_sha256']
    plan=read(pin(a.baseline/'baseline-plan.json'));pool=read(pin(a.baseline/'pool/report.json',closed['pool_sha256']))
    old_seeds=set(parent['statistics']['train']['seeds'])|set(parent['statistics']['development']['seeds'])
    splits={s:('train' if slot==0 else 'development') for t in plan['tasks'] for slot,s in enumerate(t['seeds'])}
    assert len(splits)==16 and not set(splits)&old_seeds
    contexts=[];coverage=[]
    for job in sorted(pool['jobs'],key=lambda j:j['seed']):
        task=plan['tasks'][job['task_index']];assert task['split']=='train' and job['seed'] in task['seeds']
        episode=task['episode']['id'];m=re.search(r'campaign-(.+)-site-(\d+)-',episode);assert m
        geometry=m.group(1)+'-site-'+m.group(2);folder=a.baseline/'processing'/f"case-{job['job']:04d}"
        meta=read(pin(folder/'cuda/report.json'));verification=read(pin(folder/'cuda/cuda-verification.json'))
        assert verification['state']=='passed' and verification['device']=='cuda'
        rows={r['frame']:r for r in map(json.loads,pin(folder/'cuda/rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig'))}
        windows={}
        for step in map(json.loads,pin(folder/'native/replay-0/steps.jsonl').open(encoding='utf-8-sig')):
            o=step['observation'];n=step.get('native_step');d=step.get('server_execution') or {}
            if (step['owner']=='provider' and step.get('sample') and n and o['identity']['life']==1 and o['health']>0
                    and d.get('matched') and d.get('window_exclusive') and d.get('recovery_commands')==0):
                key=(n['spawncount'],n['begin_frame'],n['actor'],n['sequence']);assert key not in windows;windows[key]=step
        first=min(s['observation']['identity']['frame'] for s in windows.values());cutoff=first+10
        capture=read(pin(Path(job['root'])/'report.json'));assert capture['capture_complete'] and capture['provenance_valid']
        log=pin(Path(capture['results'][0]['root'])/'server.log');byframe={}
        with log.open(encoding='utf-8-sig') as stream:
            for q in scan(stream,set(windows)):
                if q['eligible']:byframe.setdefault(windows[q['window']]['observation']['identity']['frame'],[]).append(q)
        candidates=[]
        for frame,shots in sorted(byframe.items()):
            if frame<cutoff or len(shots)!=1 or frame not in rows:continue
            event=rows[frame]['sample'].get('aim_candidate')
            if not event or event['candidate']!=0:continue
            q=shots[0];label=outcome(q)
            if label not in ('solid_contact','live_monster_damage'):continue
            candidates.append(dict(frame=frame,baseline_outcome=label,speed=q['metrics']['speed'],recoil_pitch=q['metrics']['recoil_pitch'],
                preferred_moving_recoil=q['metrics']['speed']>=50 and abs(q['metrics']['recoil_pitch'])>=1))
        selected=[]
        for label in ('solid_contact','live_monster_damage'):
            available=[c for c in candidates if c['baseline_outcome']==label and all(abs(c['frame']-s['frame'])>=5 for s in selected)]
            if available:selected.append(min(available,key=lambda c:(not c['preferred_moving_recoil'],c['frame'])))
        coverage.append(dict(seed=job['seed'],episode=episode,split=splits[job['seed']],first_provider_frame=first,minimum_frame=cutoff,
            eligible_late_shot_frames=len(candidates),selected=len(selected),outcomes=dict(collections.Counter(c['baseline_outcome'] for c in candidates))))
        for c in selected:
            contexts.append(dict(c,context_id=f"late-{job['seed']}-{c['frame']}",seed=job['seed'],episode=episode,geometry=geometry,job=job['job'],
                split=splits[job['seed']],offset=spec['offset']+task['seeds'].index(job['seed']),
                instance=task['instances'][task['seeds'].index(job['seed'])]))
    assert contexts and len(contexts)<=32 and len({(c['seed'],c['frame']) for c in contexts})==len(contexts)
    for p,h in pins.items():assert sha(Path(p))==h,p
    a.out.mkdir(parents=True);report=dict(state='complete',version='combat_late_aim_selection_v1',baseline=str(a.baseline.resolve()),
        parent_dataset=str(a.parent_dataset.resolve()),source=str(source.resolve()),source_weights_sha256=seal['weights_sha256'],registry=spec['registry'],
        protocol=str(a.protocol.resolve()),protocol_sha256=sha(a.protocol),contexts=contexts,coverage=coverage,new_seed_splits=splits,
        old_heldout_seeds=parent['heldout_seeds'],heldout_seeds=sorted(set(parent['heldout_seeds'])|{s for s,split in splits.items() if split=='development'}),
        inputs=pins,script_sha256=sha(__file__),training_performed=False,validation_used=False,new_games=False,promotion=False,
        scope='Fresh registry TRAIN with explicit internal train/development groups. Selection before alternative labels; no early frames substituted for missing later states.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(json.dumps(dict(contexts=len(contexts),train=sum(c['split']=='train' for c in contexts),development=sum(c['split']=='development' for c in contexts),
        moving_recoil=sum(c['preferred_moving_recoil'] for c in contexts),covered_seeds=len({c['seed'] for c in contexts}),coverage=coverage),indent=2))


if __name__=='__main__':main()
