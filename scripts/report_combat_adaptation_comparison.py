"""Summarize sealed native adaptation evaluation by policy RNG arm; no NN."""
import argparse
import pathlib
import sys

from process_combat_architecture_pool import read, save, sha, run


def compare(before, after):
    left = {(r['episode'], r['seed']): r for r in before}
    right = {(r['episode'], r['seed']): r for r in after}
    assert len(left)==len(before) and len(right)==len(after) and left.keys()==right.keys()
    return dict(episodes=len(left),
                gained_wins=sum(right[k]['win'] and not left[k]['win'] for k in left),
                lost_wins=sum(left[k]['win'] and not right[k]['win'] for k in left),
                death_delta=sum(right[k]['death']-left[k]['death'] for k in left),
                mean_received_damage_delta=sum(right[k]['received_damage']-left[k]['received_damage'] for k in left)/len(left))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=pathlib.Path,required=True); args=ap.parse_args()
    root=args.root.resolve(); repo=pathlib.Path(__file__).resolve().parents[1]
    status=read(root/'progress.json'); assert status['stage']=='complete' and status['diagnostics_complete']
    for filename,field in [('quality-report.json','quality_report_sha256'),
                           ('diagnostics-acceptance.json','diagnostics_acceptance_sha256'),
                           ('ownership-acceptance.json','ownership_acceptance_sha256'),
                           ('outcome-behavior.json','outcome_behavior_sha256'),
                           ('storage-final.json','physical_storage_sha256')]:
        assert sha(root/filename)==status[field]
    protocol=read(root/'protocol.json'); proof=read(root/'recovery/verified-members.json')
    quality=read(root/'quality-report.json'); rows=read(root/'quality-episodes.json')
    assert proof['state']=='complete' and proof['protocol_sha256']==quality['protocol_sha256']==sha(root/'protocol.json')
    assert len(rows)==quality['episodes']==protocol['total_episodes']==len(proof['members'])
    for row in rows:
        member=next(m for m in proof['members'].values() if m['report_sha256']==row['report_sha256'])
        assert member['report_sha256']==row['report_sha256']
    if not (root/'first-shot-latency.json').exists():
        run([sys.executable,repo/'scripts/report_combat_first_shot_latency.py','--root',root],root/'first-native-shot.log')
    native=read(root/'first-shot-latency.json')
    assert native['member_proof_sha256']==sha(root/'recovery/verified-members.json')
    mg=read(root/'machinegun-hits.json'); blaster=read(root/'blaster-projectile-hits.json')
    assert mg['state']=='complete'
    seal=read(root/'diagnostics-acceptance.json')
    for filename,report in [('machinegun-hits.json',mg),('blaster-projectile-hits.json',blaster)]:
        assert report['protocol_sha256']==sha(root/'protocol.json')
        assert sha(root/filename)==seal['reports'][filename]
    behavior=read(root/'outcome-behavior.json')
    aim=read(root/'machinegun-selected-target-aim.json')
    assert aim['protocol_sha256']==sha(root/'protocol.json')
    assert sha(root/'machinegun-selected-target-aim.json')==read(root/'diagnostics-acceptance.json')['reports']['machinegun-selected-target-aim.json']
    variants=[]
    for v in quality['variants']:
        label=v['variant']; m=mg['groups'][label]; b=blaster['groups'][label]
        counts=m['counts']
        variants.append(dict(variant=label,episodes=v['episodes'],wins=v['wins'],deaths=v['deaths'],
            mean_received_damage=v['mean_received_damage'],mean_outgoing_damage=v['mean_outgoing_damage'],
            machinegun=dict(shots=counts.get('shots',0),live_monster_damage_fraction=m['live_monster_damage_fraction'],
                stationary_shot_fraction=counts.get('stationary_shots',0)/counts['shots'] if counts.get('shots') else None),
            blaster=dict(shots=b['counts'].get('shots',0),live_monster_hit_fraction=b['live_monster_hit_fraction']),
            first_native_shot=native['groups'][label],machinegun_observed_target_aim=aim['groups'][label]))
    paired={}
    entries={e['model']+'-'+e['label']:e for e in protocol['evaluations']}
    baseline=protocol['comparison_reference']
    for arm in ('stochastic-a','stochastic-b'):
        before='parent-'+arm; after='after-'+arm
        assert entries[before]['policy_sampling_seed_offset']==entries[after]['policy_sampling_seed_offset']
        get=lambda label:[r for r in rows if r['variant']==label]
        paired[arm]=dict(parent_to_after=compare(get(before),get(after)),
                        rules_to_parent=compare(get(baseline),get(before)),rules_to_after=compare(get(baseline),get(after)))
    files=['protocol.json','quality-report.json','quality-episodes.json','recovery/verified-members.json',
           'machinegun-hits.json','machinegun-selected-target-aim.json','blaster-projectile-hits.json','first-shot-latency.json','outcome-behavior.json']
    result=dict(state='complete',variants=variants,paired=paired,behavior=behavior['groups'],
        inputs_sha256={f:sha(root/f) for f in files},promotion=None,
        scope='Common reused development conditions. RNG arms are separate repeats of the same conditions, not160 independent scenes. Native weapon hit fractions use their declared alive first-life windows. First-shot timing covers instrumented Blaster/Machinegun mod1/mod4 fire. No claim of optimal actions or exact angular-error misses. No promotion.')
    save(root/'adaptation-comparison.json',result)
    percent=lambda value:'—' if value is None else f'{value:.1%}'
    lines=['# Адаптация прицеливания к движению','',result['scope'],'',
           '| Вариант | Победы | Смерти | Полученный урон | MG попадания | Blaster попадания | MG выстрелы | Blaster выстрелы |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in variants:
        lines.append(f"| {v['variant']} | {v['wins']}/{v['episodes']} | {v['deaths']} | {v['mean_received_damage']:.2f} | "
                     f"{percent(v['machinegun']['live_monster_damage_fraction'])} | {percent(v['blaster']['live_monster_hit_fraction'])} | "
                     f"{v['machinegun']['shots']} | {v['blaster']['shots']} |")
    for arm,comparisons in paired.items():
        lines+=['',f"{arm}: {comparisons['parent_to_after']}"]
    (root/'adaptation-comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({'state':'complete','report':str(root/'adaptation-comparison.md')})


if __name__=='__main__': main()
