"""Re-evaluate frozen weights with current Go movement guards on common scenes."""
import argparse
import os
import pathlib
import sys

from process_combat_architecture_pool import read, save, sha, run
from run_combat_target_refresh import compile_plan
from run_combat_sampling_evaluation import collect_compressed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--families', nargs='+', required=True)
    args = ap.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    reference = args.reference.resolve()
    out = args.out.resolve()
    assert not out.exists()
    assert read(reference/'progress.json')['stage'] == 'complete'
    assert sha(reference/'result.json') == read(reference/'progress.json')['result_sha256']
    old = next(x for x in read(reference/'evaluation/protocol.json')['evaluations'] if x['model']=='quality')
    assert sha(old['plan']) == old['plan_sha256']
    template = read(old['plan'])
    weights = pathlib.Path(template['model_path'])
    assert sha(weights) == template['model_sha256'] == old['deterministic_weights_sha256']
    out.mkdir()
    os.environ['GOCACHE'] = str(repo/'workspace/build/go-cache')
    os.environ['GOTOOLCHAIN'] = 'auto'
    save(out/'progress.json', dict(stage='preparing', promotion=None))
    try:
        compiler = out/'q2episode.exe'
        run(['go','build','-o',compiler,'./cmd/q2episode'],out/'build.log')
        plan = compile_plan(compiler,repo/'scripts/scenarios/combat-training/index.json',repo,
                            weights,out/'guard','validation',args.families,28,count=4)
        compiled = read(plan)
        compiled['policy_sampling_seed_offset'] = old['policy_sampling_seed_offset']
        save(plan,compiled)
        wanted = [x for x in template['tasks'] if x['episode']['id'] in args.families]
        assert len(wanted) == len(args.families) == len(compiled['tasks'])
        assert {x['episode']['id']:(x['seeds'],x.get('instances')) for x in wanted} == {
            x['episode']['id']:(x['seeds'],x.get('instances')) for x in compiled['tasks']}
        count = sum(len(x['seeds']) for x in compiled['tasks'])
        save(out/'protocol.json', dict(version='combat_guard_common_development_v1',
             evaluations=[dict(model='quality',label='guard',root=compiled['output_root'],
                 plan=str(plan),plan_sha256=sha(plan),source_weights_sha256=old['source_weights_sha256'],
                 deterministic_weights_sha256=sha(weights),deterministic=old['deterministic'],
                 policy_sampling_seed_offset=old['policy_sampling_seed_offset'])],
             families=args.families,total_episodes=count,episodes_per_model=count,slots=16,timescale=2,
             reference_result_sha256=sha(reference/'result.json'),
             scope='Identical frozen weights, fixtures and declared RNG seeds; current Go guard versus sealed historical guard outcomes. Different Go source generations are explicit. Development only; no promotion.'))
        save(out/'progress.json',dict(stage='collecting',episodes=count,promotion=None))
        pool = collect_compressed(repo,[plan],out)
        manifest = read(pathlib.Path(pool['jobs'][0]['root'])/'manifest.json')
        run([sys.executable,repo/'scripts/verify_combat_evaluation_members.py','--root',out,
             '--source-fingerprint',manifest['source_fingerprint'],
             '--native-fingerprint',manifest['native_source_fingerprint']],out/'verify.log')
        run([sys.executable,repo/'scripts/report_combat_architecture_evaluation.py','--root',out,
             '--member-proof',out/'recovery/verified-members.json'],out/'quality.log')
        baseline = {(x['episode'],x['seed']):x for x in read(reference/'evaluation/quality-episodes.json')
                    if x['variant']=='quality-after' and x['episode'] in args.families}
        candidate = {(x['episode'],x['seed']):x for x in read(out/'quality-episodes.json')}
        assert baseline.keys() == candidate.keys() and len(candidate)==count
        prior = read(reference/'evaluation/recovery/verified-members.json')
        for task in wanted:
            for seed in task['seeds']:
                member = pathlib.Path(old['root'])/f"case-{template['tasks'].index(task)}-learned"/f's-{seed}'
                assert sha(member/'report.json') == prior['members'][str(member)]['report_sha256']
                assert read(member/'manifest.json')['native_source_fingerprint'] == manifest['native_source_fingerprint']
        for script,name in [('report_combat_machinegun_hits.py','machinegun'),
                            ('report_combat_blaster_hits.py','blaster'),
                            ('report_combat_first_shot_latency.py','latency')]:
            run([sys.executable,repo/'scripts'/script,'--root',out],out/(name+'.log'))
        result = dict(state='complete',episodes=count,weights_sha256=sha(weights),
            control_wins=sum(x['win'] for x in baseline.values()),quality_wins=sum(x['win'] for x in candidate.values()),
            control_deaths=sum(x['death'] for x in baseline.values()),quality_deaths=sum(x['death'] for x in candidate.values()),
            gained_wins=sum(candidate[k]['win'] and not baseline[k]['win'] for k in candidate),
            lost_wins=sum(baseline[k]['win'] and not candidate[k]['win'] for k in candidate),
            quality_report_sha256=sha(out/'quality-report.json'),reference_result_sha256=sha(reference/'result.json'),
            scope=read(out/'protocol.json')['scope'])
        save(out/'result.json',result)
        save(out/'progress.json',dict(stage='complete',result_sha256=sha(out/'result.json'),promotion=None))
        print(result)
    except Exception as error:
        save(out/'progress.json',dict(stage='failed',error=str(error),promotion=None))
        raise


if __name__=='__main__': main()
