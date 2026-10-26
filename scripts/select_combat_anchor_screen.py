"""Outcome-blind selection of other-map TRAIN roots from sealed full-series data."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from process_combat_architecture_pool import read,save,sha
from report_combat_machinegun_hits import scan


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--chain',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert not a.out.exists()
    chain=read(a.chain);assert chain['state']=='passed' and len(chain['updates'])==10
    pins={str(a.chain.resolve()):sha(a.chain)};available={};counts={};examined=0
    # Later source per map first; frame/seed ordering within update. No outcome
    # (damage, death, kill, bbox ray contact or alternative result) is a filter.
    for update in reversed(chain['updates']):
        fit=Path(update['fit']);fr=read(fit/'report.json')
        request_path=next(Path(p) for p in fr['inputs'] if p.endswith('train-request.json'))
        request=read(request_path);plan_path=Path(request['plan']);plan=read(plan_path)
        assert sha(plan_path)==request['plan_sha256']
        source=Path(plan['model_path']);assert sha(source)==plan['model_sha256']
        assert not any(k.startswith('diagnostic_') for k in read(source))
        pool_path=Path(request['pool_report']);pool=read(pool_path)
        assert pool['state']=='complete' and pool['source_unchanged'] and sha(pool_path)==request['pool_report_sha256']
        pins.update({str(p):sha(p) for p in (fit/'report.json',request_path,plan_path,pool_path,source)})
        jobs=sorted(pool['jobs'],key=lambda j:j['seed']);this={}
        for j in jobs:
            task=plan['tasks'][j['task_index']];name=task['episode']['map']
            assert task['split']=='train' and not j['error']
            if name in ('mine2','waste1') or name in available:continue
            member=Path(j['root']);r=read(member/'report.json')['results'][0];native=Path(r['root'])
            step_path,log_path=native/'dataset/steps.jsonl',native/'server.log'
            # Find actual MG fire windows, but never inspect their damage in selection.
            with log_path.open(encoding='utf-8-sig') as stream:shot_windows={s['window'] for s in scan(stream,set())}
            if not shot_windows:continue
            examined+=1
            for line in step_path.open(encoding='utf-8-sig'):
                s=json.loads(line);o=s['observation'];n=s.get('native_step');sample=s.get('sample')
                if not n or not sample or o['identity']['life']!=1 or o['identity']['frame']<110 or o['health']<=0:continue
                if (n['spawncount'],n['begin_frame'],n['actor'],n['sequence']) not in shot_windows:continue
                if sample.get('aim_mode',0)!=0 or o['weapon']!='models/weapons/v_machn/tris.md2' or o.get('kick_angles_degrees') is None:continue
                e=next((e for e in o['enemies'] if e['id']==s['action'].get('target_entity') and e.get('clear_shot')),None)
                if e is None:continue
                solid=e.get('observed_solid',0);lo=-((solid>>5)&31)*8;hi=((solid>>10)&63)*8-32
                if solid in (0,31) or not solid&31 or hi<=lo:continue
                f=o['identity']['frame'];seed=j['seed']
                candidate=dict(context_id=f'{name}-{seed}-{f}',map=name,seed=seed,branch_frame=f,update=update['update'],
                    offset=seed-task['episode']['splits']['train']['start'],episode_id=task['episode']['id'],
                    original_plan=str(plan_path),original_plan_sha256=sha(plan_path),source_model=str(source),source_model_sha256=sha(source),
                    baseline_member=str(member),baseline_report_sha256=sha(member/'report.json'),
                    original_steps=str(step_path),original_steps_sha256=sha(step_path))
                if name not in this or (f,seed)<(this[name]['branch_frame'],this[name]['seed']):this[name]=candidate
                break
        available.update(this);counts[str(update['update'])]=len(this)
    order=sorted(available,key=lambda name:hashlib.sha256(('anchor-other-map-screen-20261026:'+name).encode()).hexdigest())
    selected=[available[name] for name in order[:8]];assert len(selected)>=4,'Insufficient other-map roots'
    for p,h in pins.items():assert sha(p)==h
    save(a.out,dict(state='prepared',version='combat_observed_anchor_other_maps_screen_v1',contexts=selected,available_contexts=available,
        candidate_maps=len(available),by_latest_update=counts,examined_mg_captures=examined,pins=pins,
        selection='Exclude mine2/waste1. Latest sealed TRAIN update with an eligible root per other map; earliest frame>=110, then seed. Actual current MG shot, alive first life, sampled coarse, selected observed visible bbox and known current kick. Deterministic SHA map ordering, cap8. No damage/outcome/ray-contact filtering.',
        training_performed=False,promotion=False,quality_confirmation=False,
        scope='Reused TRAIN diagnostic states from potentially different frozen source models. Not fresh independent quality comparison, not ordinary PPO data.'))
    print(json.dumps(dict(maps=len(available),selected=[(c['map'],c['update'],c['branch_frame']) for c in selected])))


if __name__=='__main__':main()
