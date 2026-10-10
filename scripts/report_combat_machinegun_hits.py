"""Read-only machinegun diagnostics; native truth never becomes policy input."""
import argparse,collections,json,math,pathlib,re
from process_combat_architecture_pool import read,save,sha
from native_projectile_window import records

FIRE=re.compile(r'^sv_test_hitscan spawncount=(-?\d+) server_frame=(\d+) g_test_hitscan version=1 event=fire (.*)$')
DAMAGE=re.compile(r'^sv_test_damage spawncount=(-?\d+) server_frame=(\d+) g_test_damage version=1 (.*)$')

def fields(text):
    result={}
    for word in text.split():
        key,value=word.split('=',1)
        assert key not in result,'Duplicate telemetry field'
        result[key]=value
    return result

def vector(value,count=3):
    values=tuple(map(float,value.split(',')))
    assert len(values)==count and all(math.isfinite(v) for v in values)
    return values

def validate_shot(f):
    required='map frame shot actor mod start aim view recoil velocity spread nominal water muzzle_blocked fraction impact sky target target_class damageable health_before burst gunframe ammo_before damage'.split()
    assert set(f)==set(required),'Missing or unexpected hitscan fields'
    assert int(f['frame'])>=0 and int(f['damage'])>0 and int(f['target'])>=-1
    int(f['health_before']);int(f['damageable'])
    assert int(f['mod'])==4 and int(f['shot'])>0 and int(f['actor'])>0
    start=vector(f['start']);impact=vector(f['impact']);aim=vector(f['aim'])
    view=vector(f['view']);kick=vector(f['recoil']);velocity=vector(f['velocity'])
    pitch,yaw=map(math.radians,(view[0]+kick[0],view[1]+kick[1]))
    expected=(math.cos(pitch)*math.cos(yaw),math.cos(pitch)*math.sin(yaw),-math.sin(pitch))
    # %.6g native formatting rounds angles as well as the normalized vector.
    assert max(abs(a-b) for a,b in zip(aim,expected))<5e-5,'Aim/recoil direction mismatch'
    assert abs(sum(v*v for v in aim)-1)<1e-5
    assert 0<=float(f['fraction'])<=1 and int(f['ammo_before'])>0
    assert int(f['gunframe']) in (4,5) and 0<=int(f['burst'])<=9
    for name in ('water','muzzle_blocked','sky'):assert int(f[name]) in (0,1)
    spread=vector(f['spread'],2);nominal=vector(f['nominal'],2)
    assert all(n>=0 and abs(s)<=n*(2 if int(f['water']) else 1)+.001 for s,n in zip(spread,nominal))
    if int(f['muzzle_blocked']):assert spread==(0,0)
    return dict(speed=math.sqrt(sum(v*v for v in velocity)),recoil_pitch=kick[0],impact_distance=math.sqrt(sum((a-b)**2 for a,b in zip(start,impact))))

def scan(stream,eligible):
    shots=[];ids=set();pending=None
    for window,line in records(stream):
        if pending and window!=pending['window']:pending=None
        if line.startswith('sv_test_hitscan '):
            match=FIRE.fullmatch(line);assert match,'Malformed or truncated hitscan telemetry'
            generation,frame,tail=match.groups();f=fields(tail);metrics=validate_shot(f)
            key=(int(generation),int(f['shot']));assert key not in ids;ids.add(key)
            pending=dict(fields=f,metrics=metrics,window=window,generation=int(generation),server_frame=int(frame),damage=None,eligible=window in eligible and window is not None and int(f['actor'])==window[2])
            shots.append(pending)
        match=DAMAGE.fullmatch(line)
        if match:
            generation,frame,tail=match.groups();d=fields(tail)
            if int(d['mod'])==4 and window in eligible and window is not None and int(d['attacker'])==window[2]:
                assert pending is not None,'Native MG damage without preceding fire telemetry'
            if int(d['mod'])==4 and pending and int(d['attacker'])==int(pending['fields']['actor']):
                f=pending['fields']
                assert pending['damage'] is None and window==pending['window'] and window is not None
                assert int(generation)==pending['generation'] and int(frame)==pending['server_frame']
                assert d['map']==f['map'] and d['frame']==f['frame']
                assert int(generation)==window[0] and int(d['inflictor'])==int(f['actor'])
                assert int(d['target'])==int(f['target']) and d['health_before']==f['health_before']
                assert int(f['damageable']) and not int(f['sky']) and float(f['fraction'])<1
                pending['damage']=d
    return shots

def summarize(shots):
    counts=collections.Counter();pitch=[];speeds=[];bands=collections.defaultdict(collections.Counter)
    for shot in shots:
        if not shot['eligible']:continue
        f=shot['fields'];d=shot['damage'];m=shot['metrics'];counts['shots']+=1
        pitch.append(m['recoil_pitch']);speeds.append(m['speed'])
        counts['moving_shots' if m['speed']>=10 else 'stationary_shots']+=1
        if int(f['muzzle_blocked']):counts['muzzle_blocked']+=1
        if int(f['water']):counts['water_shots']+=1
        if int(f['sky']):outcome='sky'
        elif float(f['fraction'])>=1:outcome='no_contact'
        elif not int(f['damageable']):outcome='solid_contact'
        elif int(f['health_before'])<=0:outcome='corpse_contact'
        elif d and int(d['health_after'])<int(d['health_before']):
            outcome='live_monster_damage' if f['target_class'].startswith('monster_') else 'other_damage'
            counts['health_damage']+=int(d['health_before'])-int(d['health_after'])
        else:outcome='damageable_contact_without_health_damage'
        counts[outcome]+=1
        labels=['moving' if m['speed']>=10 else 'stationary',
            'recoil_0_to_1.5' if abs(m['recoil_pitch'])<=1.5 else 'recoil_1.5_to_6' if abs(m['recoil_pitch'])<=6 else 'recoil_above_6']
        for label in labels:
            bands[label]['shots']+=1;bands[label][outcome]+=1
    total=counts['shots']
    return dict(counts=dict(counts),live_monster_damage_fraction=counts['live_monster_damage']/total if total else None,
        mean_recoil_pitch=sum(pitch)/total if total else None,mean_native_speed=sum(speeds)/total if total else None,
        descriptive_strata={label:dict(counts=dict(c),live_monster_damage_fraction=c['live_monster_damage']/c['shots']) for label,c in bands.items()})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);a=ap.parse_args()
    protocol=read(a.root/'protocol.json');proof=read(a.root/'recovery/verified-members.json')
    assert proof['state']=='complete' and proof['protocol_sha256']==sha(a.root/'protocol.json')
    groups={};sources=[]
    for entry in protocol['evaluations']:
        aggregate=[]
        for path in pathlib.Path(entry['root']).glob('case-*/s-*/report.json'):
            report=read(path);assert report['capture_complete'] and report['provenance_valid']
            assert proof['members'][str(path.parent)]['report_sha256']==sha(path)
            root=pathlib.Path(report['results'][0]['root']);steps=root/'dataset/steps.jsonl';log=root/'server.log';eligible=set()
            with steps.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    step=json.loads(line);o=step['observation'];n=step.get('native_step')
                    if n and o['identity']['life']==1 and o['identity']['frame']>100 and o['health']>0 and step.get('server_execution',{}).get('matched'):
                        eligible.add((n['spawncount'],n['begin_frame'],n['actor'],n['sequence']))
            with log.open(encoding='utf-8-sig') as stream:shots=scan(stream,eligible)
            aggregate.extend(shots);sources.append(dict(server_log=str(log),server_sha256=sha(log),steps_sha256=sha(steps),telemetry_shots=len(shots)))
        if protocol['version']=='native_machinegun_smoke_v1':
            assert aggregate,'Smoke arm must exercise machinegun telemetry'
        groups[entry['model']+'-'+entry['label']]=summarize(aggregate)
    save(a.root/'machinegun-hits.json',dict(version='native_machinegun_ordered_window_v1',state='complete',protocol_sha256=sha(a.root/'protocol.json'),groups=groups,sources=sources,
        scope='Actual machinegun fire traces and synchronous damage joined by ordered command window, actor and target. First-life alive matched dispatch only. Contact without native health damage remains explicit. Recoil direction reconstructed without NN execution. No selected-target credit, no training inputs, no model promotion.'))
    print(json.dumps(groups),flush=True)

if __name__=='__main__':main()
