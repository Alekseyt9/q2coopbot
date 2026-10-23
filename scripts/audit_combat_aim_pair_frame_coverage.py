"""Measure TRAIN shot coverage lost by constant-head record-high gating."""
import argparse
import collections
import json
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha
from report_combat_machinegun_hits import scan
from report_registered_machinegun_outcomes import outcome
from combat_aim_candidates import frame_uniforms


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('baseline','dataset','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    closed=read(pin(a.baseline/'complete.json'));pool=read(pin(a.baseline/'pool/report.json',closed['pool_sha256']))
    dsseal=read(pin(a.dataset/'complete.json'));dataset=read(pin(a.dataset/'report.json',dsseal['report_sha256']))
    heldout=set(dataset['heldout_seeds']);cases=[];smoke=[]
    for job in pool['jobs']:
        root=a.baseline/'processing'/f"case-{job['job']:04d}";meta=read(pin(root/'cuda/report.json'))
        rows=list(map(json.loads,pin(root/'cuda/rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig')))
        steps=list(map(json.loads,pin(root/'native/replay-0/steps.jsonl').open(encoding='utf-8-sig')));windows={}
        for s in steps:
            o=s['observation'];n=s.get('native_step');d=s.get('server_execution') or {}
            if s['owner']=='provider' and s.get('sample') and n and o['identity']['life']==1 and o['health']>0 and d.get('matched') and d.get('window_exclusive') and d.get('recovery_commands')==0:
                windows[(n['spawncount'],n['begin_frame'],n['actor'],n['sequence'])]=s
        capture=read(pin(Path(job['root'])/'report.json'));log=pin(Path(capture['results'][0]['root'])/'server.log');shots={}
        with log.open(encoding='utf-8-sig') as stream:
            for q in scan(stream,set(windows)):
                if q['eligible']:shots.setdefault(windows[q['window']]['observation']['identity']['frame'],[]).append(q)
        maximum=0.;eligible=0;reachable=0;counts=collections.Counter()
        for r in rows:
            event=r['sample'].get('aim_candidate')
            if not event:continue
            gate,_=frame_uniforms(r['sample']['sampling_seed'],r['frame'],1);record=gate-maximum>=1e-5;maximum=max(gate,maximum)
            if len(shots.get(r['frame'],[]))!=1:continue
            q=shots[r['frame']][0];label=outcome(q)
            if label not in ('live_monster_damage','solid_contact'):continue
            eligible+=1;reachable+=int(record);counts[label]+=1
            if (not record and job['seed'] not in heldout and r['frame']>=110 and label=='solid_contact'
                    and q['metrics']['speed']>10 and abs(q['metrics']['recoil_pitch'])>.25 and r['sample'].get('aim_mode',0)==1):
                smoke.append(dict(seed=job['seed'],frame=r['frame'],job=job['job'],speed=q['metrics']['speed'],
                    recoil_pitch=q['metrics']['recoil_pitch'],baseline_outcome=label,constant_gate_cannot_first_intervene=True,
                    source_sample=r['sample'],episode=capture['results'][0]['generated_fixture']['episode_id']))
        cases.append(dict(seed=job['seed'],split='development' if job['seed'] in heldout else 'train',
            actual_eligible_shot_frames=eligible,constant_head_reachable_frames=reachable,outcomes=dict(counts)))
    total=sum(c['actual_eligible_shot_frames'] for c in cases);reachable=sum(c['constant_head_reachable_frames'] for c in cases)
    smoke.sort(key=lambda c:(not c['episode'].startswith('campaign-ware2-site-01'),c['seed'],c['frame']))
    for p,h in pins.items():assert sha(Path(p))==h,p
    a.out.mkdir(parents=True);report=dict(state='complete',version='combat_aim_pair_frame_coverage_v1',cases=cases,
        eligible_native_shot_frames=total,constant_head_reachable_frames=reachable,coverage=reachable/total if total else None,
        smoke_candidate=smoke[0] if smoke else None,inputs=pins,script_sha256=sha(__file__),new_games=False,
        training_performed=False,neural_execution=False,promotion=False,
        scope='Read-only actual TRAIN MG window coverage. Record-high gate is a limitation of current diagnostic constant-head collection, not a statement about policy capacity. No development labels become TRAIN.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(json.dumps(dict(eligible_native_shot_frames=total,constant_head_reachable_frames=reachable,coverage=report['coverage'],
        smoke_candidate={k:v for k,v in smoke[0].items() if k!='source_sample'} if smoke else None),indent=2))


if __name__=='__main__':main()
