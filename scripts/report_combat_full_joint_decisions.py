"""Describe native decision traces; counts are not shot accuracy or causation."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from process_combat_architecture_pool import read, save, sha
from audit_registered_damage_timing import damage_events


def trace(member):
    result = read(member/'report.json')['results'][0]
    path = Path(result['root'])/'bot.jsonl'
    rows = [json.loads(s) for s in path.open(encoding='utf-8-sig')]
    life = result['first_life']
    rows = [r for r in rows if life['start_frame'] <= r.get('observation_frame',-1) <= life['end_frame'] and r.get('health',0)>0]
    counts = Counter(); timeline = []; prior_target = None; prior_attack = None
    for r in rows:
        c = r.get('combat_policy') or {}; sel = c.get('selection') or {}
        obs = c.get('observation') or {}
        candidate = sel.get('candidate') or {}; applied = c.get('applied') or {}
        sample = sel.get('sample') or {}
        counts['observed_alive_frames'] += 1
        counts['provider_frames'] += int(sel.get('owner') == 'provider')
        counts['applied_attack_frames'] += int(bool(applied.get('attack')))
        if not sample: continue
        counts['sample_frames'] += 1
        clear = [e for e in obs['enemies'] if e.get('clear_shot')]
        counts['clear_frames'] += int(bool(clear))
        counts['clear_no_selected_target'] += int(bool(clear) and not candidate.get('target_entity'))
        counts['clear_attack_requested'] += int(bool(clear) and candidate['attack'])
        counts['clear_attack_blocked'] += int(bool(clear) and candidate['attack'] and not applied.get('attack'))
        counts['jump_requested'] += int(candidate['vertical']=='jump')
        counts['stationary_frames'] += int(math.hypot(*obs['velocity'][:2])<40)
        counts['yaw_over_10_degrees'] += int(abs(candidate['yaw_delta_degrees'])>10)
        counts['pitch_over_10_degrees'] += int(abs(candidate['pitch_delta_degrees'])>10)
        counts['weapon_request_frames'] += int(bool(candidate.get('weapon')))
        target = candidate.get('target_entity')
        counts['target_switches'] += int(prior_target is not None and target != prior_target)
        item = dict(frame=obs['identity']['frame'],target=target,attack=candidate['attack'],
            applied_attack=bool(applied.get('attack')),vertical=candidate['vertical'],
            yaw=candidate['yaw_delta_degrees'],pitch=candidate['pitch_delta_degrees'],
            weapon=obs['weapon'],ammo=obs['ammo'],gun_frame=obs['gun_frame'],health=obs['health'],
            position=obs['position'],velocity=obs['velocity'],latent=sample['latent'],
            requested_weapon=candidate.get('weapon'),clear_targets=[e['id'] for e in clear])
        timeline.append(item); prior_target=target; prior_attack=candidate['attack']
    return result,dict(counts),timeline,path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evaluation',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists()
    evaluation=read(a.evaluation/'validation-summary.json')
    assert evaluation['state']=='complete' and evaluation['paired_conditions_exact']
    records=[];pins={}
    for case in evaluation['cases']:
        member=Path(case['capture_root'])
        result,counts,timeline,path=trace(member)
        pins[str(path)]=sha(path);pins[str(member/'report.json')]=sha(member/'report.json')
        root=Path(result['root']);events=damage_events((root/'server.log').read_text(encoding='utf-8-sig').splitlines())
        life=result['first_life'];events=[e for e in events if e['spawncount']==life['spawncount'] and e['map']==life['map'] and life['start_frame']<=e['frame']<=life['end_frame']]
        outgoing=[e for e in events if e['attacker']==life['bot_entity'] and e['target_class'].startswith('monster_')]
        records.append(dict(arm=case['arm'],map=case['map'],seed=case['seed'],goal=case['goal'],death=case['death'],
            counts=counts,first_sample=timeline[:1],last_samples=timeline[-8:],
            first_damage_frame=min((e['frame'] for e in outgoing),default=None),
            damage_by_target={str(t):sum(max(0,e['health_before'])-max(0,e['health_after']) for e in outgoing if e['target']==t) for t in {e['target'] for e in outgoing}},
            native_kill_frames=[dict(frame=e['frame'],target=e['target']) for e in outgoing if e['health_before']>0 and e['health_after']<=0],
            timeline=timeline))
        pins[str(root/'server.log')]=sha(root/'server.log')
    totals={}
    for arm in ('before','after','rules'):
        counts=Counter()
        for record in records:
            if record['arm']==arm:counts.update(record['counts'])
        totals[arm]=dict(counts)
    pair={r['arm']:r for r in records if r['seed']==1919052130}
    assert set(pair)=={'before','after','rules'}
    before={r['frame']:r for r in pair['before']['timeline']}
    after={r['frame']:r for r in pair['after']['timeline']}
    firsts={}
    for field in ('target','attack','vertical','requested_weapon','position'):
        for frame in sorted(before.keys()&after.keys()):
            if before[frame][field]!=after[frame][field]:
                firsts[field]=dict(frame=frame,before=before[frame][field],after=after[frame][field]);break
    save(a.out,dict(state='complete',totals=totals,lost_space_first_differences=firsts,
        cases=records,inputs=pins,scope='Observed traces and native damage only; no NN inference or causal attribution. Attack-frame counts are not shots. Rules absent sample metadata is not no-target evidence.'))
    print(json.dumps(dict(totals=totals,lost_space_first_differences=firsts)))


if __name__=='__main__':main()
