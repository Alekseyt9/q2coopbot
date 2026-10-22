"""Actual learned-provider MG outcomes, including shots without selected targets."""
import argparse
import collections
import json
import pathlib
from process_combat_architecture_pool import read,save,sha
from report_combat_machinegun_hits import scan
from report_combat_machinegun_aim import measure


def outcome(shot):
    f,d=shot['fields'],shot['damage']
    if int(f['sky']):return 'sky'
    if float(f['fraction'])>=1:return 'no_contact'
    if not int(f['damageable']):return 'solid_contact'
    if int(f['health_before'])<=0:return 'corpse_contact'
    if d and int(d['health_after'])<int(d['health_before']):
        return 'live_monster_damage' if f['target_class'].startswith('monster_') else 'other_damage'
    return 'damageable_without_health_damage'


def summary(shots):
    counts=collections.Counter(outcome(s) for s in shots)
    count=len(shots)
    return dict(shots=count,outcomes=dict(counts),
        live_monster_damage_fraction=counts['live_monster_damage']/count if count else None,
        instantaneous_non_damage_ray_count=sum(counts[k] for k in ('sky','no_contact','solid_contact','corpse_contact')),
        muzzle_blocked=sum(int(s['fields']['muzzle_blocked']) for s in shots))


def main():
    ap=argparse.ArgumentParser()
    for name in ('evaluation','spec','aim','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    args=ap.parse_args();assert not args.out.exists()
    root=args.evaluation.resolve();pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path);digest=sha(path);assert expected is None or expected==digest
        pins[str(path.resolve())]=digest;return path
    closed=read(pin(root/'complete.json'));assert closed['state']=='complete' and not closed['training_performed']
    spec=read(pin(args.spec,closed['spec_sha256']))
    pool=read(pin(root/'pool/report.json',closed['pool_sha256']));assert pool['state']=='complete' and pool['source_unchanged']
    aim=read(pin(args.aim));assert aim['state']=='complete' and not aim['training_performed']
    for item in aim['inputs']:pin(item['path'],item['sha256'])
    for name in ('report_combat_machinegun_hits.py','report_combat_machinegun_aim.py','native_projectile_window.py'):
        pin(pathlib.Path(__file__).with_name(name))
    jobs={j['job']:j for j in pool['jobs']};variants={}
    for index,variant in enumerate(spec['variants']):
        if variant['mode']!='learned':continue
        label=variant['label'];execution=read(pin(root/('execution-'+label+'.json')))
        assert execution['state']=='complete' and execution['pool_sha256']==closed['pool_sha256']
        assert execution['selected_plan_index']==index
        bindings={c['job']:c for c in execution['cases']};aggregate=[];cases=[]
        for case in aim['variants'][label]['cases']:
            job=jobs[case['job']];binding=bindings[job['job']];assert job['plan_index']==index and not job['error']
            capture=read(pin(pathlib.Path(job['root'])/'report.json',binding['capture_sha256']))
            log=pin(pathlib.Path(capture['results'][0]['root'])/'server.log',case['server_log_sha256'])
            steps=pin(root/('processing-'+label)/f"case-{job['job']:04d}/native/replay-0/steps.jsonl",case['steps_sha256'])
            windows={}
            with steps.open(encoding='utf-8-sig') as stream:
                for step in map(json.loads,stream):
                    o=step['observation'];native=step.get('native_step');dispatch=step.get('server_execution') or {}
                    if (not native or step['owner']!='provider' or o['identity']['life']!=1 or o['health']<=0
                            or o['identity']['frame']<binding['network_release_frame'] or not dispatch.get('matched')
                            or not dispatch.get('window_exclusive') or dispatch.get('recovery_commands')!=0):continue
                    assert native['spawncount']==binding['release_generation']
                    key=(native['spawncount'],native['begin_frame'],native['actor'],native['sequence'])
                    assert key not in windows;windows[key]=step
            with log.open(encoding='utf-8-sig') as stream:shots=[s for s in scan(stream,set(windows)) if s['eligible']]
            statuses=collections.Counter()
            for shot in shots:
                status=measure(shot,windows[shot['window']])['status'];shot['target_status']=status;statuses[status]+=1
            assert len(shots)==case['eligible_shots'] and dict(statuses)==case['statuses']
            cases.append(dict(job=job['job'],seed=job['seed'],episode=case['episode'],all=summary(shots),
                by_target_status={s:summary([q for q in shots if q['target_status']==s]) for s in sorted(statuses)}))
            aggregate.extend(shots)
        assert len(cases)==len(bindings)
        variants[label]=dict(cases=cases,all=summary(aggregate),
            by_target_status={s:summary([q for q in aggregate if q['target_status']==s]) for s in sorted({q['target_status'] for q in aggregate})})
    for path,digest in pins.items():assert sha(path)==digest
    save(args.out,dict(state='complete',version='registered_actual_machinegun_outcomes_v1',variants=variants,
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),
        training_performed=False,neural_execution=False,promotion=False,
        scope='Actual MG fires in exclusive unrecovered first-life alive provider windows from canonical native network release. Same denominator/statuses as sealed selected-aim report, including undeclared targets. Living monster damage may concern a different enemy. Non-damage rays describe immediate outcome, not strategic avoidability; armor/protection contacts remain ambiguous. Muzzle-blocked is overlapping. Rules and other weapons excluded; no estimated shot rate from attack commands, hidden policy inputs or fitting.'))
    print(json.dumps({k:dict(all=v['all'],by_target_status=v['by_target_status']) for k,v in variants.items()}))


if __name__=='__main__':main()
