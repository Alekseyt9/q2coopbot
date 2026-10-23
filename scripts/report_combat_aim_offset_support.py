"""Describe a fixed-muzzle/observed-bbox proxy for the 25 aim offsets; no NN."""
import argparse
import collections
import json
import math
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from report_combat_machinegun_hits import scan,vector
from report_combat_selected_target_aim import ray


def intersects(start,direction,lower,upper):
    entry,exit=0.,float('inf')
    for p,d,lo,hi in zip(start,direction,lower,upper):
        if abs(d)<1e-12:
            if p<lo or p>hi:return False
        else:
            a,b=(lo-p)/d,(hi-p)/d
            entry=max(entry,min(a,b));exit=min(exit,max(a,b))
    return exit>=entry


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('baseline','selection','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    seal=read(pin(a.baseline/'complete.json'));pool=read(pin(a.baseline/'pool/report.json',seal['pool_sha256']))
    ss=read(pin(a.selection/'complete.json'));selection=read(pin(a.selection/'report.json',ss['report_sha256']))
    assert selection['baseline']==str(a.baseline.resolve()) and pool['source_unchanged']
    selected={(c['seed'],c['frame']) for c in selection['contexts']};records=[]
    for job in sorted(pool['jobs'],key=lambda j:j['job']):
        folder=a.baseline/'processing'/f"case-{job['job']:04d}"/'native/replay-0';windows={}
        with pin(folder/'steps.jsonl').open(encoding='utf-8-sig') as stream:
            for s in map(json.loads,stream):
                o=s['observation'];n=s.get('native_step');d=s.get('server_execution') or {}
                if s['owner']=='provider' and s.get('sample') and n and o['identity']['life']==1 and o['health']>0 and d.get('matched') and d.get('window_exclusive') and d.get('recovery_commands')==0:
                    windows[n['spawncount'],n['begin_frame'],n['actor'],n['sequence']]=s
        capture=read(pin(Path(job['root'])/'report.json'));log=pin(Path(capture['results'][0]['root'])/'server.log')
        with log.open(encoding='utf-8-sig') as stream:shots=[q for q in scan(stream,set(windows)) if q['eligible']]
        for q in shots:
            s=windows[q['window']];o=s['observation'];frame=o['identity']['frame'];action=s['action'];sample=s['sample']
            entity=action.get('target_entity',0);track=action.get('target_track',0)
            r=dict(seed=job['seed'],frame=frame,selected_context=(job['seed'],frame) in selected,status='unknown_selected_bbox')
            e=next((e for e in o['enemies'] if entity and e['id']==entity and e.get('observed_track',0)==track),None)
            event=sample.get('aim_candidate')
            if not e or not e.get('clear_shot') or not event:records.append(r);continue
            solid=e.get('observed_solid',0);radius=(solid&31)*8;bottom=-((solid>>5)&31)*8;top=((solid>>10)&63)*8-32
            if solid in (0,31) or not radius or top<=bottom:records.append(r);continue
            origin=[o['position'][i]+e['relative'][i] for i in range(3)]
            lower=[origin[0]-radius,origin[1]-radius,origin[2]+bottom];upper=[origin[0]+radius,origin[1]+radius,origin[2]+top]
            start=vector(q['fields']['start']);view=vector(q['fields']['view']);kick=vector(q['fields']['recoil'])
            z=event['source_latent'];current=sample['latent'];scale=15. if sample.get('aim_mode',0)==1 else 180.
            support=[];offsets=(0.,-5.,-2.5,2.5,5.)
            for k in range(25):
                delta=[]
                for axis,off in ((2,offsets[k//5]),(3,offsets[k%5])):
                    ratio=math.tanh(z[axis]) if k==0 else max(-1+1e-6,min(1-1e-6,(math.tanh(z[axis])*scale+off)/scale))
                    delta.append(ratio*scale-math.tanh(current[axis])*scale)
                direction=ray(view[1]+kick[1]+delta[0],view[0]+kick[0]+delta[1])
                if intersects(start,direction,lower,upper):support.append(k)
            damage=q['damage'];hit=bool(damage and int(damage['target'])==entity and int(damage['health_before'])>0 and int(damage['health_after'])<int(damage['health_before']))
            r.update(status='measured',proxy_supported_categories=support,proxy_baseline_supported=event['candidate'] in support,
                     actual_selected_target_damage=hit,native_target_class=q['fields']['target_class'])
            records.append(r)
    statistics={}
    for name,group in (('all',records),('selected',[r for r in records if r['selected_context']])):
        measured=[r for r in group if r['status']=='measured']
        statistics[name]=dict(shots=len(group),statuses=dict(collections.Counter(r['status'] for r in group)),
            proxy_any_supported=sum(bool(r['proxy_supported_categories']) for r in measured),
            proxy_none_supported=sum(not r['proxy_supported_categories'] for r in measured),
            actual_selected_target_damage=sum(r['actual_selected_target_damage'] for r in measured),
            proxy_none_despite_actual_selected_damage=sum(not r['proxy_supported_categories'] and r['actual_selected_target_damage'] for r in measured),
            proxy_baseline_unsupported_despite_actual_selected_damage=sum(not r['proxy_baseline_supported'] and r['actual_selected_target_damage'] for r in measured))
    for p,h in pins.items():assert sha(p)==h,p
    a.out.mkdir(parents=True);path=a.out/'measurements.jsonl';path.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in records),encoding='utf-8')
    save(a.out/'report.json',dict(state='complete',version='combat_observed_bbox_offset_support_proxy_v1',statistics=statistics,
         inputs=pins,script_sha256=sha(__file__),measurements_sha256=sha(path),training_performed=False,new_games=0,
         scope='Diagnostic proxy only: fixed actual native muzzle, observed pre-command bbox, changed native input angles. Ignores world obstruction, motion response, target displacement, integer-degree quantization and spread. Not a native counterfactual oracle, fire mask, expected hit rate or learning input.'))
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(json.dumps(statistics,indent=2))


if __name__=='__main__':main()
