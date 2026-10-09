"""Actual first-life blaster projectile outcomes, with unknown endings explicit."""
import argparse,json,pathlib,re,collections
from process_combat_architecture_pool import read,sha,save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);a=ap.parse_args()
    protocol=read(a.root/'protocol.json');proof=read(a.root/'recovery/verified-members.json')
    assert proof['state']=='complete' and proof['protocol_sha256']==sha(a.root/'protocol.json')
    pattern=re.compile(r'^sv_test_projectile spawncount=(-?\d+) server_frame=(\d+) g_test_projectile version=1 event=(spawn|end) map=\w+ frame=\d+ shot=(\d+) entity=(\d+) (.*)$')
    damage=re.compile(r'^sv_test_damage spawncount=(-?\d+) server_frame=(\d+) g_test_damage version=1 map=\w+ frame=\d+ (.*)$')
    groups={};sources=[]
    for entry in protocol['evaluations']:
        counts=collections.Counter()
        for path in pathlib.Path(entry['root']).glob('case-*/s-*/report.json'):
            report=read(path);assert report['capture_complete'] and report['provenance_valid']
            assert proof['members'][str(path.parent)]['report_sha256']==sha(path)
            root=pathlib.Path(report['results'][0]['root']);steps=root/'dataset/steps.jsonl';log=root/'server.log';valid=set()
            with steps.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    s=json.loads(line);o=s['observation'];n=s.get('native_step')
                    if n and o['identity']['life']==1 and o['identity']['frame']>100 and o['health']>0 and s.get('server_execution',{}).get('matched'):
                        valid.add((n['spawncount'],n['begin_frame'],n['actor']))
            shots={};contacts={};ready=False
            with log.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    line=line.strip()
                    if line=='g_test_projectile ready version=1':ready=True
                    match=pattern.match(line)
                    if match:
                        generation,frame,event,shot,entity,tail=match.groups();generation,frame,shot,entity=map(int,(generation,frame,shot,entity));key=(generation,shot);fields=dict(v.split('=',1) for v in tail.split())
                        if event=='spawn':
                            assert key not in shots
                            shots[key]=dict(entity=entity,attacker=int(fields['attacker']),mod=int(fields['mod']),frame=frame,eligible=(generation,frame,int(fields['attacker'])) in valid,outcome='unresolved',target=0)
                        else:
                            assert key in shots and shots[key]['outcome']=='unresolved' and shots[key]['entity']==entity
                            shots[key].update(outcome=fields['outcome'],target=int(fields['target']),end_frame=frame)
                    match=damage.match(line)
                    if match:
                        generation,frame,tail=match.groups();fields=dict(v.split('=',1) for v in tail.split());shot=int(fields.get('shot',0))
                        if shot:
                            key=(int(generation),shot);assert key not in contacts;contacts[key]=fields
            assert ready
            for key,s in shots.items():
                if not s['eligible'] or s['mod']!=1:continue
                counts['shots']+=1;counts[s['outcome']]+=1;contact=contacts.get(key)
                if contact:
                    assert int(contact['attacker'])==s['attacker'] and int(contact['inflictor'])==s['entity'] and int(contact['mod'])==1
                    if s['outcome']=='damage':assert int(contact['target'])==s['target']
                    else:assert s['outcome']=='unresolved'
                    live=int(contact['health_before'])>0 and int(contact['health_after'])<int(contact['health_before'])
                    if live and contact['target_class'].startswith('monster_'):counts['live_monster_hits']+=1
                    elif int(contact['health_before'])<=0:counts['corpse_contacts']+=1
                elif s['outcome']=='damage':raise AssertionError('Damage ending without native contact')
            sources.append(dict(server_log=str(log),server_sha256=sha(log),steps_sha256=sha(steps)))
        counts['unknown']=counts['unresolved']+counts['freed'];shots=counts['shots'];hits=counts['live_monster_hits'];unknown=counts['unknown']
        groups[entry['model']+'-'+entry['label']]=dict(counts=counts,live_monster_hit_fraction=hits/shots if shots and not unknown else None,confirmed_hit_fraction_lower=hits/shots if shots else None,possible_hit_fraction_upper=min(1,(hits+unknown)/shots) if shots else None)
    save(a.root/'blaster-projectile-hits.json',dict(protocol_sha256=sha(a.root/'protocol.json'),groups=groups,sources=sources,scope='Actual native mod1 projectile IDs joined to first-life alive dispatch after frame100. Outcomes may close after firing window. Unknown unresolved/freed remain explicit; definitive fraction only when none unknown. No machinegun denominator, no selected-target credit, no policy observation from server data.'))
    print(json.dumps(groups),flush=True)

if __name__=='__main__':main()
