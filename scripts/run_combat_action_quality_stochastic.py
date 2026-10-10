"""Two common stochastic RNG arms, including groups, for reward-v6 A/B actors."""
import argparse
import os
import pathlib
import shutil
import sys

from process_combat_architecture_pool import read, save, sha, run
from run_combat_target_refresh import compile_plan
from run_combat_sampling_evaluation import collect_compressed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--training',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    args = ap.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    training, out = args.training.resolve(), args.out.resolve()
    assert read(training/'progress.json')['stage'] == 'complete' and not out.exists()
    assert shutil.disk_usage(out.parent).free > 8*1024**3
    out.mkdir()
    os.environ['GOCACHE'] = str(repo/'workspace/build/go-cache')
    os.environ['GOTOOLCHAIN'] = 'auto'
    evaluation = out/'evaluation'
    evaluation.mkdir()
    families = ['parasite-gunner-blaster-generated','parasite-gunner-machinegun-recoil',
                'campaign-base1-site-02-blaster','campaign-base1-site-02-machinegun',
                'campaign-base2-site-02-blaster','campaign-base2-site-02-machinegun']
    entries, plans, common = [], [], None
    try:
        for name in ('control','quality'):
            update = training/name/'processing'/name/'update'
            seal = read(update/'complete.json')
            assert sha(update/'weights.json') == seal['weights_sha256']
            for label, offset in [('stochastic-a',20261011),('stochastic-b',20261012)]:
                model = read(update/'weights.json')
                model['deterministic'] = False
                weights = evaluation/(name+'-'+label+'-weights.json')
                save(weights,model)
                path = compile_plan(training/'q2episode.exe',repo/'scripts/scenarios/combat-training/index.json',
                                    repo,weights,evaluation/(name+'-'+label),'validation',families,28,count=4)
                plan = read(path)
                plan['policy_sampling_seed_offset'] = offset
                save(path,plan)
                conditions = {t['episode']['id']:(t['seeds'],t.get('instances')) for t in plan['tasks']}
                if common is None:
                    common = conditions
                assert common == conditions
                plans.append(path)
                entries.append(dict(model=name,label=label,root=plan['output_root'],plan=str(path),
                    plan_sha256=sha(path),source_weights_sha256=sha(update/'weights.json'),
                    deterministic_weights_sha256=sha(weights),deterministic=False,policy_sampling_seed_offset=offset))
        save(evaluation/'protocol.json',dict(version='combat_action_quality_stochastic_ab_v1',
             evaluations=entries,families=families,total_episodes=96,episodes_per_model=24,slots=16,timescale=2,
             training_result_sha256=sha(training/'result.json'),
             scope='Reused development validation28, two distinct common policy RNG offsets. Includes parasite/gunner groups '
                   'not trained in the small reward A/B. Same geometry/engine seeds per arm; no independent final test or promotion.'))
        save(out/'progress.json',dict(stage='paired_validation',episodes=96,promotion=None))
        collect_compressed(repo,plans,evaluation)
        manifest = read(pathlib.Path(read(evaluation/'pool/report.json')['jobs'][0]['root'])/'manifest.json')
        run([sys.executable,repo/'scripts/verify_combat_evaluation_members.py','--root',evaluation,
             '--source-fingerprint',manifest['source_fingerprint'],'--native-fingerprint',manifest['native_source_fingerprint']],out/'verify.log')
        run([sys.executable,repo/'scripts/audit_combat_sampling_configs.py','--root',evaluation],out/'sampling-config-audit.log')
        assert not read(evaluation/'sampling-config-audit.json')['repeated_stochastic_execution_configs']
        save(out/'progress.json',dict(stage='captures_complete',validation_episodes=96,promotion=None))
        run([sys.executable,repo/'scripts/report_combat_action_quality_ab.py','--root',out],out/'report.log')
    except Exception as error:
        save(out/'progress.json',dict(stage='failed',error=str(error),promotion=None))
        raise


if __name__ == '__main__':
    main()
