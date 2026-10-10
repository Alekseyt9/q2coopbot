"""Compare a sealed reward candidate to its unchanged parent and rules; no training."""
import argparse
import pathlib
import shutil
import sys

from combat_native_evaluation_closure import finalize
from process_combat_architecture_pool import read, save, sha, run
from run_combat_ppo_series import sealed_update
from run_combat_sampling_evaluation import collect_compressed
from run_combat_stochastic_heldout import verify_sources
from run_combat_target_refresh import compile_plan


def main():
    ap = argparse.ArgumentParser()
    for name in ('reference','parent-update','candidate-update','out'):
        ap.add_argument('--'+name, type=pathlib.Path, required=True)
    a = ap.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    reference, parent, candidate, out = [p.resolve() for p in
        (a.reference,a.parent_update,a.candidate_update,a.out)]
    assert not out.exists()
    status = read(reference/'progress.json')
    assert status['stage'] == 'complete' and status['result_sha256'] == sha(reference/'result.json')
    result = read(reference/'result.json')
    assert result['state'] == 'complete' and result['ownership_clean']
    for file,digest in result['native_reports_sha256'].items():
        assert sha(reference/'evaluation'/file) == digest
    assert result['quality_report_sha256'] == sha(reference/'evaluation/quality-report.json')
    spec = read(reference/'protocol.json')
    assert pathlib.Path(spec['parent']).resolve() == parent
    assert sha(parent/'weights.json') == spec['parent_weights_sha256']
    assert candidate == reference/'quality/processing/quality/update'
    before, after = sealed_update(parent), sealed_update(candidate)
    assert after['updates_completed'] == before['updates_completed']+1
    assert before['architecture'] == after['architecture']
    source_binding = verify_sources(repo,reference/'evaluation')
    assert shutil.disk_usage(out.parent).free > 8*1024**3
    out.mkdir()
    compiler = reference/'q2episode.exe'
    template = read(reference/'quality/capture/train/plan.json')
    assert len(spec['families']) == len(set(spec['families'])) == 20
    save(out/'selection.json',dict(reference=str(reference),reference_result_sha256=sha(reference/'result.json'),
        parent_update=str(parent),candidate_update=str(candidate),
        parent_checkpoint_sha256=sha(parent/'checkpoint.pt'),candidate_checkpoint_sha256=sha(candidate/'checkpoint.pt'),
        compiler_sha256=sha(compiler),registry_path=template['registry_path'],registry_sha256=template['registry_sha256'],
        scope='Development candidate follow-up, not final test or promotion. Reward-v9 candidate versus unchanged strong parent and rules.'))
    try:
        plans, entries, conditions = [], [], None
        for name, update in [('parent',parent),('after',candidate),('rules',None)]:
            weights = None
            if update:
                weights = out/(name+'-weights.json')
                shutil.copyfile(update/'weights.json',weights)
                assert not read(weights)['deterministic']
            for label,offset in ([('stochastic-a',20261011),('stochastic-b',20261012)] if weights else [('baseline',0)]):
                path = compile_plan(compiler,template['registry_path'],repo,weights,
                                    out/(name+'-'+label),'validation',spec['families'],28)
                plan = read(path)
                plan['policy_sampling_seed_offset'] = offset
                save(path,plan)
                current = {t['episode']['id']:(t['seeds'],t['instances']) for t in plan['tasks']}
                if conditions is None:
                    conditions = current
                assert current == conditions
                assert sum(len(t['seeds']) for t in plan['tasks']) == 80
                run([compiler,'--verify-plan',path,'--root',repo],path.parent/'verify.log')
                plans.append(path)
                entries.append(dict(model=name,label=label,root=plan['output_root'],plan=str(path),plan_sha256=sha(path),
                    source_weights_sha256=sha(weights) if weights else None,
                    deterministic_weights_sha256=sha(weights) if weights else None,
                    deterministic=False if weights else True,policy_sampling_seed_offset=offset))
        save(out/'protocol.json',dict(version='combat_native_candidate_development_v1',evaluations=entries,
            families=spec['families'],episodes_per_model=80,total_episodes=400,slots=16,timescale=2,
            comparison_reference='rules-baseline',reference_result_sha256=sha(reference/'result.json'),
            validation_offset=28,training_device='cuda',promotion=None,
            scope='Parent and candidate x two policy RNG arms plus rules. Identical80 generated validation conditions28..31; '
                  'RNG arms repeat these scenes, not160 independent conditions. Common v9 labels; reward sums are not quality metrics. '
                  'Reused development, reserved test untouched; no training or automatic promotion.'))
        assert verify_sources(repo,reference/'evaluation') == source_binding
        save(out/'progress.json',dict(stage='evaluating',episodes=400,promotion=None))
        pool = collect_compressed(repo,plans,out)
        finalize(repo,out,pool)
        run([sys.executable,repo/'scripts/report_combat_adaptation_comparison.py','--root',out],out/'comparison.log')
        run([sys.executable,repo/'scripts/audit_combat_machinegun_waste.py','--root',out],out/'machinegun-waste.log')
        run([sys.executable,repo/'scripts/audit_combat_miss_reward.py','--root',out,'--min-frame','100'],out/'blaster-miss-credit.log')
        progress = read(out/'progress.json')
        progress.update(comparison_sha256=sha(out/'adaptation-comparison.json'),
                        machinegun_waste_sha256=sha(out/'machinegun-waste-audit.json'),
                        blaster_miss_credit_sha256=sha(out/'miss-reward-audit-after-frame-100.json'))
        save(out/'progress.json',progress)
    except Exception as error:
        save(out/'progress.json',dict(stage='failed',error=str(error),promotion=None))
        raise


if __name__ == '__main__':
    main()
