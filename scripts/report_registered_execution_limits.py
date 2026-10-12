"""Separate intervention counts from observed standing in sealed evaluations."""
import argparse,collections,pathlib
from process_combat_architecture_pool import read,save,sha


def aggregate(cases):
    execution=sum((collections.Counter(c['execution_counts']) for c in cases),collections.Counter())
    behavior=sum((collections.Counter(c['behavior_counts']) for c in cases),collections.Counter())
    reasons=sum((collections.Counter(c['reasons']) for c in cases),collections.Counter())
    def fraction(n,d):return n/d if d else None
    return dict(cases=len(cases),wins=sum(c['goal'] for c in cases),execution_counts=dict(execution),
        behavior_counts=dict(behavior),reasons=dict(reasons),
        movement_suppressed_fraction=fraction(execution['movement_command_fully_suppressed_frames'],execution['proposed_movement_frames']),
        attack_suppressed_fraction=fraction(execution['attack_suppressed_frames'],execution['proposed_attack_frames']),
        stationary_clear_fraction=fraction(behavior['stationary_clear_enemy_frames'],execution['clear_enemy_frames']))


def main():
    ap=argparse.ArgumentParser()
    for name in ('evaluation','spec','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    args=ap.parse_args();assert not args.out.exists()
    root=args.evaluation.resolve();seal_path=root/'complete.json';seal=read(seal_path)
    assert seal['state']=='complete' and not seal['training_performed'] and not seal['promotion']
    assert sha(args.spec)==seal['spec_sha256'];spec=read(args.spec)
    pool_path=root/'pool/report.json';assert sha(pool_path)==seal['pool_sha256']
    pool=read(pool_path);assert pool['state']=='complete' and pool['source_unchanged']
    assert pool['usable_captures']==len(pool['jobs'])==seal['native_cases']
    pins={seal_path:sha(seal_path),args.spec:seal['spec_sha256'],pool_path:seal['pool_sha256']}
    variants={}
    for index,v in enumerate(spec['variants']):
        if v['mode']!='learned':continue
        label=v['label'];ep=root/('execution-'+label+'.json');bp=root/('behavior-'+label+'.json')
        execution,behavior=read(ep),read(bp);pins[ep]=sha(ep);pins[bp]=sha(bp)
        assert execution['state']=='complete' and execution['version']=='executed_action_intervention_audit_v2'
        assert behavior['version']=='registered_target_intent_audit_v4'
        assert execution['pool_sha256']==behavior['pool_sha256']==seal['pool_sha256']
        assert execution['selected_plan_index']==behavior['selected_plan_index']==index
        jobs={j['job']:j for j in pool['jobs'] if j['plan_index']==index}
        bc={c['job']:c for c in behavior['cases']};assert len(bc)==len(behavior['cases'])==len(jobs)
        cases=[];seen=set()
        for c in execution['cases']:
            assert c['job'] not in seen;seen.add(c['job']);b=bc[c['job']];job=jobs[c['job']]
            assert not job['error'] and job['mode']=='learned'
            for field in ('seed','episode','trace_sha256','capture_sha256','manifest_sha256','server_log_sha256','release_generation','native_release_game_frame'):
                assert c[field]==b[field],field
            assert c['seed']==job['seed'] and c['network_release_frame']==b['release_frame']
            assert c['counts']['provider_frames']==b['counts']['provider_frames']
            assert c['counts']['applied_attack_frames']==b['counts'].get('attack_command_frames',0)
            assert c['counts']['movement_command_fully_suppressed_frames']>=c['counts']['airborne_movement_suppressed_frames']
            capture_path=pathlib.Path(job['root'])/'report.json';assert sha(capture_path)==c['capture_sha256']
            cases.append(dict(job=c['job'],seed=c['seed'],episode=c['episode'],goal=b['goal'],
                trace_sha256=c['trace_sha256'],capture_sha256=c['capture_sha256'],
                execution_counts=c['counts'],behavior_counts=b['counts'],reasons=c['reasons']))
        assert len(cases)==len(jobs)
        groups={}
        for name in ('all','ordinary','recovery'):
            rows=[c for c in cases if name=='all' or c['episode'].endswith('-recovery')==(name=='recovery')]
            if rows:groups[name]=aggregate(rows)
        variants[label]=dict(cases=cases,groups=groups)
    for p,d in pins.items():assert sha(p)==d,str(p)
    save(args.out,dict(state='complete',version='registered_execution_limits_report_v1',variants=variants,
        inputs=[dict(path=str(p),sha256=d) for p,d in pins.items()],script_sha256=sha(pathlib.Path(__file__)),
        training_performed=False,neural_execution=False,promotion=False,
        scope='Sealed correct-network-clock first-life provider audits joined by identical capture, trace, release and counts. Standing uses pre-command observed XY speed below10. Suppression fractions use proposed nonzero commands. Group sums do not establish frame overlap or causality: suppressed acceleration can coexist with moving bodies, applied motion with wall collisions. Grounded suppression is total minus airborne; it does not identify a particular reason. Reasons may overlap. No new trace replay, fitting, rewards or physical-shot claim.'))
    print('Execution limits report:',len(variants),'variants',flush=True)


if __name__=='__main__':main()
