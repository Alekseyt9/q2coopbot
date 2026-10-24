"""Align proposed/executed movement with its actual next native observation."""
import argparse
import collections
import json
import math
from pathlib import Path
import numpy as np
from evaluate_registered_combat_models import read,save,sha

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('cache','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or h==expected,str(path);pins[str(path)]=h;return path
    pin(__file__);closed=read(pin(a.cache/'complete.json'));cache=read(pin(a.cache/'report.json',closed['report_sha256']))
    assert cache['version']=='combat_action_inertia_source_cache_v1' and len(cache['chunks'])==24
    total=collections.Counter();maps={};cases=[]
    for item in cache['chunks']:
        assert item['split']=='train'
        with np.load(pin(item['path'],item['sha256'])) as d:ids=set(map(int,d['step']))
        folder=Path(item['case']['folder']);meta=read(pin(folder/'cuda/report.json'));sp=next((folder/'native').glob('replay-*/steps.jsonl'));pin(sp,meta['source_sha256'][str(sp.resolve())])
        counts=collections.Counter();reasons=collections.Counter()
        with sp.open(encoding='utf-8-sig') as stream:
            for s in map(json.loads,stream):
                if s['index'] not in ids:continue
                o=s['observation'];n=s['next_observation'];assert n and n['identity']['frame']==o['identity']['frame']+1
                request=s['action'];applied=s['applied_action']
                rn=math.hypot(request['forward'],request['side']);an=math.hypot(applied['forward'],applied['side'])
                displacement=math.hypot(*(y-x for x,y in zip(o['position'][:2],n['position'][:2])))
                current=math.hypot(*o['velocity'][:2]);next_speed=math.hypot(*n['velocity'][:2])
                visible_ground=bool(o['enemies'] and o['on_ground']);large=rn>=.25;sent=an>=.25
                counts['rows']+=1;counts['nontrivial_request_rows']+=large;counts['nontrivial_applied_rows']+=sent
                counts['request_reduced_to_near_zero_rows']+=bool(large and an<.1)
                counts['visible_ground_rows']+=visible_ground
                counts['standing_before_nontrivial_request_rows']+=bool(visible_ground and current<40 and large)
                counts['low_speed_after_nontrivial_applied_rows']+=bool(visible_ground and next_speed<40 and sent)
                counts['near_zero_displacement_after_nontrivial_applied_rows']+=bool(visible_ground and displacement<1 and sent)
                counts['near_zero_displacement_after_nontrivial_applied_release_rows']+=bool(visible_ground and displacement<1 and sent and applied['vertical']=='release')
                for v in s.get('interventions') or []:
                    if v.get('component')=='movement':reasons[v['reason']]+=1
        assert counts['rows']==item['rows'];total.update(counts);mp=item['case']['map'];maps.setdefault(mp,collections.Counter()).update(counts)
        cases.append(dict(seed=item['case']['seed'],map=mp,counts=dict(counts),recorded_movement_interventions=dict(reasons)))
    assert total['rows']==cache['rows']
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True);save(a.out/'report.json',dict(state='complete',version='combat_movement_native_step_alignment_v1',counts=dict(total),maps={k:dict(v) for k,v in maps.items()},cases=cases,inputs=pins,
        neural_execution=False,new_games=False,training_performed=False,promotion=False,
        scope='TRAIN-only descriptive observed movement alignment. Current speed precedes current command; actual next speed/displacement follows it. Large request does not imply executed movement, and low travel alone does not prove a bug or useless action. No hidden state or new reward.'))
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')));print(dict(total))

if __name__=='__main__':main()
