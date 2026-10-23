"""Explain duration-screen exclusions from immutable captures; never admit or fit."""
import argparse
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from audit_combat_joint_aim_grid import physical_equal_with_entity_bijection


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collection',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();root=a.collection.resolve();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or h==expected,str(path);pins[str(path)]=h;return path
    closed=read(pin(root/'complete.json'));spec=read(pin(root/'spec.json',closed['spec_sha256']))
    p=read(pin(spec['protocol'],spec['protocol_sha256']))
    dc=read(pin(root/'dataset/complete.json'));report=read(pin(root/'dataset/report.json',dc['report_sha256']))
    previous=[]
    for item in p['previous_collections']:
        old=Path(item['collection']);cs=read(pin(old/'complete.json'));os=read(pin(old/'spec.json',cs['spec_sha256']))
        previous.extend((old,u) for u in os['contexts'])
    def load(collection,unit,label):
        v=next(v for v in unit['variants'] if v['label']==label);i=v['plan_index'];folder=collection/f'case-{i:02d}'
        meta=read(pin(folder/'cuda/report.json'));step=next((folder/'native').glob('replay-*/steps.jsonl'))
        allsteps=list(map(json.loads,pin(step,meta['source_sha256'][str(step.resolve())]).open(encoding='utf-8-sig')))
        pool=read(pin(collection/'pool/report.json'));job=next(j for j in pool['jobs'] if j['plan_index']==i)
        capture=read(pin(Path(job['root'])/'report.json'));result=capture['results'][0]
        steps=[s for s in allsteps if s['owner']=='provider' and s.get('sample') and s.get('native_step') and s['observation']['identity']['life']==1 and s['observation']['health']>0]
        return dict(steps=steps,allsteps=allsteps,result=result,case=str(folder))
    def first_diff(a,b,path=''):
        if type(a)!=type(b):return dict(path=path,left=a,right=b)
        if isinstance(a,dict):
            for k in sorted(set(a)|set(b)):
                if k not in a or k not in b:return dict(path=path+'/'+k,left=a.get(k),right=b.get(k))
                d=first_diff(a[k],b[k],path+'/'+k)
                if d:return d
        elif isinstance(a,list):
            if len(a)!=len(b):return dict(path=path+'/length',left=len(a),right=len(b))
            for i,(x,y) in enumerate(zip(a,b)):
                d=first_diff(x,y,path+'/'+str(i))
                if d:return d
        elif a!=b:return dict(path=path,left=a,right=b)
        return None
    findings=[]
    for exclusion in report['excluded']:
        cid,seed=exclusion['context_id'],exclusion['stream_seed']
        unit=next(u for u in spec['contexts'] if u['context']['context_id']==cid and read(u['prepared']+'/report.json')['fork']['seed']==seed)
        if exclusion['reason'].startswith('One/five'):
            old,ou=next((old,u) for old,u in previous if u['context']['context_id']==cid and read(u['prepared']+'/report.json')['fork']['seed']==seed)
            left,right=load(old,ou,'control'),load(root,unit,'control');forward={};reverse={};first=None
            for x,y in zip(left['steps'],right['steps']):
                frame=x['observation']['identity']['frame']
                if frame!=y['observation']['identity']['frame']:first=dict(frame=frame,reason='Different provider frame');break
                for key in ('observation','next_observation'):
                    xx,yy=x.get(key),y.get(key)
                    same=(xx is None and yy is None) or (xx is not None and yy is not None and physical_equal_with_entity_bijection(xx,yy,forward,reverse))
                    if not same:first=dict(frame=frame,reason=key,first_raw_diff=first_diff(xx,yy));break
                if first:break
                fields=('sampling_seed','latent','attack','vertical','weapon','target','aim_mode','memory','aim_candidate','continuation_probe')
                for key in fields:
                    if x['sample'].get(key,0)!=y['sample'].get(key,0):first=dict(frame=frame,reason='sample/'+key,first_raw_diff=first_diff(x['sample'].get(key,0),y['sample'].get(key,0)));break
                if first:break
                if x['sent_command']!=y['sent_command']:first=dict(frame=frame,reason='command',first_raw_diff=first_diff(x['sent_command'],y['sent_command']));break
            summary=lambda d:dict(provider_rows=len(d['steps']),first_provider_frame=d['steps'][0]['observation']['identity']['frame'],
                last_provider_frame=d['steps'][-1]['observation']['identity']['frame'],goal=d['result']['goal_stop'],death=d['result']['death_stop'])
            findings.append(dict(exclusion=exclusion,left=summary(left),right=summary(right),first_difference=first))
        else:
            proof=read(pin(root/'proofs'/unit['unit_id']/'report.json'))
            for result in proof['results']:
                if result['admitted']:continue
                data=load(root,unit,result['label']);end=data['result']['goal_stop']['kill_frame'] if data['result']['goal_stop'] else data['result']['first_life']['end_frame']
                rules=[s for s in data['allsteps'] if s['owner']=='rules' and unit['context']['frame']<=s['observation']['identity']['frame']<end and s['observation']['health']>0]
                bot=Path(data['result']['root'])/'bot.jsonl';events=[]
                if bot.exists():
                    pin(bot)
                    for line in bot.open(encoding='utf-8-sig'):
                        obj=json.loads(line)
                        selection=(obj.get('combat_policy') or {}).get('selection') or {}
                        if selection.get('owner')=='rules' and obj.get('frame') in {s['observation']['identity']['frame'] for s in rules}:
                            events.append(dict(frame=obj['frame'],owner=selection['owner'],fallback=selection.get('fallback')))
                findings.append(dict(exclusion=exclusion,label=result['label'],case=data['case'],rules_frames=[s['observation']['identity']['frame'] for s in rules],events=events,
                    handoff_observations=[dict(frame=s['observation']['identity']['frame'],visible_enemies=len(s['observation']['enemies']),remembered_threats=len(s['observation'].get('remembered_threats',[]))) for s in rules[:2]]))
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True);save(a.out/'report.json',dict(state='complete',version='combat_window_exclusion_audit_v1',findings=findings,inputs=pins,
        neural_execution=False,new_games=False,admission_changed=False,training_performed=False,promotion=False))
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(json.dumps(findings,ensure_ascii=False))


if __name__=='__main__':main()
