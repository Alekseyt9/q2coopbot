"""Extend finite observation without changing the original short-horizon record."""
import argparse
import json
from pathlib import Path

from audit_combat_corpse_motion_probe import run as base_run
from audit_combat_checked_ground_native_probe import normalized, same
from process_combat_architecture_pool import read, save, sha


def run(protocol, root, out, short):
    spec=read(protocol)
    assert spec['cases'][0]['game_frames']==500
    short_report=read(short/'report.json')
    short_proof=read(short/'analysis/complete.json')
    assert short_proof['state']=='complete' and sha(short/'analysis/report.json')==short_proof['report_sha256']
    base_run(protocol,root,out)
    pool=read(root/'report.json')
    pins={str(Path(__file__).resolve()):sha(Path(__file__)),str(short/'analysis/report.json'):sha(short/'analysis/report.json')}

    def trace(job):
        capture=read(Path(job['root'])/'report.json')
        path=Path(capture['results'][0]['root'])/'bot.jsonl'
        pins[str(path)]=sha(path)
        result={}
        for line in path.open(encoding='utf-8-sig'):
            raw=json.loads(line);o=raw.get('combat_policy',{}).get('observation',{})
            if o.get('identity',{}).get('life')==1 and o.get('health',0)>0:
                result[o['identity']['frame']]=raw
        return result,capture['results'][0]

    checks, details={},[]
    for job in pool['jobs']:
        old=next(j for j in short_report['jobs'] if j['label']==job['label'] and j['repeat']==job['repeat'])
        previous,_=trace(old);current,row=trace(job)
        frames=[f for f in previous if 98<=f<=398]
        def signature(raw):
            c=raw['combat_policy']
            return normalized({k:c[k] for k in ('observation','proposed','proposed_command','applied_command')})
        key=job['label']+'/'+str(job['repeat'])
        checks[key]=all(f in current and same(signature(previous[f]),signature(current[f]),1e-9) for f in frames)
        assert checks[key], 'Horizon extension changed original past'
        effects_path=Path(row['root'])/'dataset/server_outcomes.jsonl'
        pins[str(effects_path)]=sha(effects_path)
        damage=[event for line in effects_path.open(encoding='utf-8-sig') for event in json.loads(line).get('events',[])
                if event['target']==327 and event['health_before']<=0]
        observed_dead_frames=[f for f,raw in current.items() if any(b['id']==327 and b['observed_death_animation']
            for b in (raw['combat_policy']['observation'].get('observed_dynamic_bodies') or {}).get('bodies',[]))]
        details.append(dict(label=job['label'],repeat=job['repeat'],verified_prefix_frames=len(frames),
            last_observed_dead_body_frame=max(observed_dead_frames,default=None),
            corpse_native_health_after=damage[-1]['health_after'] if damage else None,
            corpse_native_last_damage_frame=damage[-1]['frame'] if damage else None,
            corpse_reached_native_gib_threshold=any(e['health_after']<=-130 for e in damage)))
    assert all(sha(Path(p))==digest for p,digest in pins.items())
    base=read(out/'report.json')
    save(out/'long-report.json',dict(state='complete',checks=checks,details=details,records=base['records'],inputs=pins,
        short_report_sha256=sha(short/'analysis/report.json'),long_report_sha256=sha(out/'report.json'),
        scope='Separate500-frame diagnostic; exact original prefixes through398 for every arm/repeat. Same world and RNG; no independent quality claim or optimal-action label. Corpse threshold from native medic source; observed body disappearance alone is not gib proof.'))
    save(out/'long-complete.json',dict(state='complete',report_sha256=sha(out/'long-report.json')))
    print(json.dumps(details,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for k in ('protocol','root','out','short'):
        parser.add_argument('--'+k,required=True,type=Path)
    args=parser.parse_args()
    run(args.protocol.resolve(),args.root.resolve(),args.out.resolve(),args.short.resolve())
