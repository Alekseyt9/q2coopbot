"""Join actual MG shots to observed geometry and stored PPO reward components.

Read-only TRAIN audit. No policy inference, hypothetical rewards or refitting.
"""
import argparse
import collections
import json
import math
import pathlib
from process_combat_architecture_pool import read,save,sha
from native_combat_release import release_boundary
from report_combat_machinegun_hits import scan
from report_registered_machinegun_outcomes import outcome


def known_bbox(enemy):
    solid=enemy.get('observed_solid')
    return bool(enemy.get('clear_shot') and solid is not None and solid not in (0,31)
        and solid&31 and ((solid>>10)&63)*8-32>-((solid>>5)&31)*8)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--corpus',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True)
    args=ap.parse_args();assert not args.out.exists();root=args.corpus.resolve();pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path);digest=sha(path);assert expected is None or digest==expected
        pins[str(path.resolve())]=digest;return path
    seal=read(pin(root/'complete.json'));assert seal['state']=='complete' and not seal['training_performed']
    protocol=read(pin(root/'protocol.json',seal['protocol_sha256']))
    assert protocol['train_only'] and protocol['ppo_training_objective'] and not protocol['diagnostic_intervention']
    pool=read(pin(root/'pool/report.json',seal['pool_sha256']));assert pool['state']=='complete' and pool['source_unchanged']
    assert len(seal['processed'])==1 and seal['processed'][0]['label']=='train'
    processed_path=pin(seal['processed'][0]['report'],seal['processed'][0]['sha256'])
    processing=processed_path.parent;processed=read(processed_path)
    assert processed['state']=='complete' and processed['device']=='cuda' and not processed['training_performed']
    assert all(c['split']=='train' for c in processed['cases'])
    merged=read(pin(processing/'merged/report.json'));assert merged['reward_config_sha256']==protocol['ppo_training_objective']['reward_sha256']
    reward_paths={i['path'] for i in protocol['inputs'] if i['sha256']==merged['reward_config_sha256']}
    assert len(reward_paths)==1
    reward_config=read(pin(next(iter(reward_paths)),merged['reward_config_sha256']))
    native_cost_verified=0;native_cost_total=0.
    with pin(processing/'merged/rollout.jsonl',merged['rollout_sha256']).open(encoding='utf-8-sig') as stream:
        rows={(r['seed'],r['index']):r for r in map(json.loads,stream)}
    assert len(rows)==merged['rows'];jobs={j['job']:j for j in pool['jobs']};buckets=collections.defaultdict(list);cases=[]
    for item in processed['cases']:
        job=jobs[item['job']];capture=read(pin(pathlib.Path(job['root'])/'report.json'));assert capture['capture_complete'] and capture['provenance_valid']
        manifest=read(pin(pathlib.Path(job['root'])/'manifest.json'))
        log=pin(pathlib.Path(capture['results'][0]['root'])/'server.log')
        with log.open(encoding='utf-8-sig') as stream:generation,release,_=release_boundary(stream,manifest,job['seed'])
        native=processing/f"case-{job['job']:04d}/native/replay-0"
        with pin(native/'rewards.jsonl').open(encoding='utf-8-sig') as stream:rewards={r['step']:r for r in map(json.loads,stream)}
        windows={}
        with pin(native/'steps.jsonl').open(encoding='utf-8-sig') as stream:
            for step in map(json.loads,stream):
                if (job['seed'],step['index']) not in rows:continue
                o=step['observation'];n=step.get('native_step');d=step.get('server_execution') or {}
                if (not n or step['owner']!='provider' or o['identity']['life']!=1 or o['health']<=0 or o['identity']['frame']<release
                        or not d.get('matched') or not d.get('window_exclusive') or d.get('recovery_commands')!=0):continue
                reward=rewards[step['index']];assert reward['available']
                assert math.isclose(reward['score'],rows[(job['seed'],step['index'])]['reward'],rel_tol=1e-10,abs_tol=1e-10)
                assert n['spawncount']==generation
                key=(n['spawncount'],n['begin_frame'],n['actor'],n['sequence']);assert key not in windows
                windows[key]=(step,reward)
        with log.open(encoding='utf-8-sig') as stream:shots=[s for s in scan(stream,set(windows)) if s['eligible']]
        local=collections.Counter();misses=collections.Counter()
        for shot in shots:
            step,reward=windows[shot['window']];o=step['observation'];components=reward['components']
            assert reward['version']==merged['reward_version']
            data=dict(seed=job['seed'],index=step['index'],outcome=outcome(shot),
                known_bbox=any(known_bbox(e) for e in o['enemies']),
                direct_observed_weapon=o['weapon'] in ('Blaster','Machinegun','models/weapons/v_machn/tris.md2'),
                target_entity=step['action'].get('target_entity',0),off_target_attack=components['off_target_attack'])
            buckets['all'].append(data);local[data['outcome']]+=1
            if data['outcome'] in ('sky','no_contact','solid_contact','corpse_contact'):misses[shot['window']]+=1
            buckets['known_bbox' if data['known_bbox'] else 'no_known_bbox'].append(data)
            if not data['known_bbox'] and data['direct_observed_weapon']:buckets['no_known_bbox_direct_weapon'].append(data)
        if merged['reward_version'] in ('combat_reward_v9','combat_reward_v13'):
            for window,(_,reward) in windows.items():
                expected=misses[window]*reward_config['machinegun_miss']
                actual=reward['components']['machinegun_miss']
                assert math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-10), 'Stored native miss cost differs from actual rays'
                native_cost_verified+=1;native_cost_total+=actual
        cases.append(dict(job=job['job'],seed=job['seed'],shots=len(shots),outcomes=dict(local)))
    result={}
    for name,items in buckets.items():
        result[name]=dict(shots=len(items),command_windows=len({(i['seed'],i['index']) for i in items}),
            outcomes=dict(collections.Counter(i['outcome'] for i in items)),
            off_target_penalized_shots=sum(i['off_target_attack']<0 for i in items),
            unselected_target_shots=sum(i['target_entity']==0 for i in items))
    for name in ('native_combat_release.py','report_combat_machinegun_hits.py','report_registered_machinegun_outcomes.py'):
        pin(pathlib.Path(__file__).with_name(name))
    for p,h in pins.items():assert sha(p)==h
    save(args.out,dict(state='complete',version='registered_train_fire_reward_audit_v2',cases=cases,summary=result,
        native_miss_cost_verified_windows=native_cost_verified,native_miss_cost_total=native_cost_total,
        reward_config_sha256=merged['reward_config_sha256'],reward_version=merged['reward_version'],
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),
        training_performed=False,neural_execution=False,promotion=False,
        scope='Actual exclusive unrecovered MG fire windows present in accepted TRAIN rollout, joined to stored available reward components. Reward score equals PPO row score. Known bbox predicate follows observed clear solid bounds, not future native visibility. Counts of shots and unique command windows differ; no reward totals inferred by summing per-shot components. off_target_attack is one component, not total return. Instantaneous solid/corpse/sky outcomes do not prove strategic avoidability. No changed objective, labels, weights or runtime actions.'))
    print(json.dumps(result))


if __name__=='__main__':main()
