"""Native paired outcomes by registered loadout/map/site, with sealed members."""
import argparse,collections,json,pathlib
from process_combat_architecture_pool import read,save,sha

def summarize(rows):
    n=len(rows);assert n
    return dict(episodes=n,wins=sum(r['win'] for r in rows),deaths=sum(r['death'] for r in rows),
                mean_outgoing_damage=sum(r['outgoing_damage'] for r in rows)/n,
                mean_received_damage=sum(r['received_damage'] for r in rows)/n)

def paired(candidate,reference):
    left={(r['episode'],r['seed']):r['win'] for r in candidate}
    right={(r['episode'],r['seed']):r['win'] for r in reference}
    assert len(left)==len(candidate) and len(right)==len(reference) and left.keys()==right.keys()
    gained=sum(left[k] and not right[k] for k in left);lost=sum(right[k] and not left[k] for k in left)
    return dict(pairs=len(left),gained_wins=gained,lost_wins=lost,win_delta=gained-lost)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);a=ap.parse_args();root=a.root.resolve()
    protocol=read(root/'protocol.json');proof=read(root/'recovery/verified-members.json');quality=read(root/'quality-report.json')
    assert proof['state']=='complete' and proof['protocol_sha256']==sha(root/'protocol.json')
    assert quality['state']=='complete' and quality['protocol_sha256']==sha(root/'protocol.json') and quality['pool_sha256']==sha(root/'recovery/verified-members.json')
    rows=read(root/'quality-episodes.json');assert len(rows)==protocol['total_episodes']
    metadata={};members={}
    for entry in protocol['evaluations']:
        assert sha(entry['plan'])==entry['plan_sha256'];plan=read(entry['plan']);variant=entry['model']+'-'+entry['label']
        for index,task in enumerate(plan['tasks']):
            episode=task['episode'];family=episode['id'];item=dict(map=episode['map'],loadout=episode['recipe']['loadout'],site=family)
            if family in metadata:assert metadata[family]==item
            metadata[family]=item
            for seed in task['seeds']:
                member=pathlib.Path(entry['root'])/f"case-{index}-{task['modes'][0]}"/f's-{seed}'
                assert sha(member/'report.json')==proof['members'][str(member)]['report_sha256']
                members[(variant,family,seed)]=read(member/'report.json')['results'][0]
    assert len(members)==len(rows)
    for row in rows:
        key=(row['variant'],row['episode'],row['seed']);assert key in members
        native=members.pop(key);damage=native['first_life']['damage'];goal=native.get('goal_stop')
        assert row['win']==bool(goal and goal['reason']=='combat_goal_complete')
        assert row['death']==(native['first_life']['end_reason']=='first_observed_death')
        assert row['outgoing_damage']==sum(v.get('monster_health_damage',0) for v in damage.get('by_mod',[]))
        assert row['received_damage']==damage['received_health_damage']
    assert not members
    variants=sorted({r['variant'] for r in rows});references=[v for v in ('firebc-baseline','rules-baseline') if v in variants];groups={}
    for axis in ('loadout','map','site'):
        groups[axis]={}
        for value in sorted({m[axis] for m in metadata.values()}):
            selected=[r for r in rows if metadata[r['episode']][axis]==value]
            grouped={v:[r for r in selected if r['variant']==v] for v in variants};assert all(grouped.values())
            groups[axis][value]=dict(variants={v:summarize(items) for v,items in grouped.items()},
                comparisons=[dict(variant=v,reference=ref,**paired(items,grouped[ref])) for ref in references for v,items in grouped.items() if v!=ref])
    report=dict(protocol_sha256=sha(root/'protocol.json'),quality_sha256=sha(root/'quality-report.json'),episodes_sha256=sha(root/'quality-episodes.json'),member_proof_sha256=sha(root/'recovery/verified-members.json'),groups=groups,
        scope='Sealed first-life native outcomes. Registered loadout is initial inventory, not a claim about every fired weapon. Site conditions are isolated fights on campaign geometry, not full map completion. Development validation; no final-test superiority claim.')
    save(root/'quality-strata.json',report)
    lines=['# Native результаты по оружию, картам и условиям','',report['scope'],'']
    for axis,values in groups.items():
        lines += [f'## {axis}','','| Условие | Вариант | Победы | Смерти | Урон монстрам | Полученный урон |','| --- | --- | ---: | ---: | ---: | ---: |']
        for value,data in values.items():
            for variant,s in data['variants'].items():
                lines.append(f"| {value} | {variant} | {s['wins']}/{s['episodes']} | {s['deaths']} | {s['mean_outgoing_damage']:.1f} | {s['mean_received_damage']:.1f} |")
        lines += ['']
    (root/'quality-strata.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(episodes=len(rows),strata={axis:len(values) for axis,values in groups.items()})))

if __name__=='__main__':main()
