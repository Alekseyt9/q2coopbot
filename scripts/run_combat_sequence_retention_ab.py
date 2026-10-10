"""Alternative CUDA updates from one sealed parent/corpus, then common native evaluation."""
import argparse
import concurrent.futures
import os
import pathlib
import shutil
import sys
from combat_native_evaluation_closure import finalize
from process_combat_architecture_pool import read, save, sha, run
from run_combat_ppo_series import sealed_update
from run_combat_sampling_evaluation import collect_compressed
from run_combat_stochastic_heldout import verify_sources
from run_combat_target_refresh import compile_plan
from train_combat_bc import torch


def equal_cuda(left, right):
    if torch.is_tensor(left):
        return left.is_cuda and right.is_cuda and torch.equal(left,right)
    if isinstance(left,dict):
        return left.keys()==right.keys() and all(equal_cuda(left[k],right[k]) for k in left)
    if isinstance(left,(tuple,list)):
        return len(left)==len(right) and all(equal_cuda(a,b) for a,b in zip(left,right))
    return left==right


def main():
    parser=argparse.ArgumentParser()
    for name in ('series','reference','out'):
        parser.add_argument('--'+name,type=pathlib.Path,required=True)
    parser.add_argument('--weight',type=float,default=1.)
    args=parser.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[1]
    series,reference,out=(p.resolve() for p in (args.series,args.reference,args.out))
    assert torch.cuda.is_available() and 0 < args.weight <= 10 and not out.exists()
    execution=read(series/'progress.json')
    assert execution['stage']=='complete' and execution['completed_rounds']==1
    spec=read(series/'protocol.json')
    parent=pathlib.Path(spec['parent_update'])
    original=pathlib.Path(execution['final_update'])
    sealed_update(parent);sealed_update(original)
    assert sha(original/'weights.json')==execution['final_weights_sha256']
    closed=read(reference/'progress.json')
    assert closed['stage']=='complete' and closed['diagnostics_complete']
    assert read(reference/'analysis-progress.json')['stage']=='complete'
    assert read(reference/'ownership-acceptance.json')['clean']
    assert sha(reference/'quality-report.json')==closed['quality_report_sha256']
    source_binding=verify_sources(repo,reference)
    data=series/'round-1/processing/series/rollout'
    processing=series/'round-1/processing'
    retention=series/'sequence-retention-spec.json'
    assert read(data/'report.json')['model_sha256']==sha(parent/'weights.json')
    assert shutil.disk_usage(out.parent).free > 8*1024**3
    out.mkdir()
    frozen=out/'python-sources';frozen.mkdir()
    for path in (repo/'scripts').glob('*.py'):
        shutil.copyfile(path,frozen/path.name)
    save(out/'protocol.json',dict(version='combat_sequence_retention_ab_v1',series=str(series),
         series_protocol_sha256=sha(series/'protocol.json'),reference=str(reference),
         parent_update=str(parent),parent_weights_sha256=sha(parent/'weights.json'),
         rollout_sha256=sha(data/'rollout.jsonl'),sequence_sha256=sha(data/'sequence.jsonl'),
         retention_spec_sha256=sha(retention),weight=args.weight,device='cuda',
         python_sources={p.name:sha(p) for p in frozen.glob('*.py')},
         scope='Two alternative updates from the same original on-policy parent/corpus; no sequential reuse after actor changes. Control must reproduce the original update exactly. No validation states in training.'))
    try:
        save(out/'progress.json',dict(stage='cuda_training',promotion=None))
        def train(name,weight):
            branch=out/name;branch.mkdir()
            (branch/'rollout').mkdir()
            os.link(data/'sequence.jsonl',branch/'rollout/sequence.jsonl')
            run([sys.executable,frozen/'ppo_recurrent.py','--model',parent/'weights.json',
                 '--data',data,'--config',processing/'config.json','--out',branch/'update',
                 '--resume',parent/'checkpoint.pt','--anchor-model',processing/'anchor.json',
                 '--retention-bank',processing/'bank.json','--retention-weight','0','--bank-weight','0',
                 '--sequence-retention-spec',retention,'--sequence-retention-weight',weight,
                 '--fork-sequence-retention'],branch/'train.log')
            return branch/'update'
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            control=executor.submit(train,'control',0.)
            quality=executor.submit(train,'quality',args.weight)
            control,quality=control.result(),quality.result()
        assert sha(control/'weights.json')==sha(original/'weights.json'), 'Zero-weight control failed exact reproduction'
        old=torch.load(original/'checkpoint.pt',map_location='cuda',weights_only=True)
        replay=torch.load(control/'checkpoint.pt',map_location='cuda',weights_only=True)
        fields=('actor','value','log_std','actor_optimizer','value_optimizer','rng','cuda_rng')
        assert all(equal_cuda(old[k],replay[k]) for k in fields)
        save(out/'control-reproduction.json',dict(state='complete',device='cuda',weights_byte_exact=True,
             state_exact=list(fields),weights_sha256=sha(control/'weights.json')))
        run([sys.executable,repo/'scripts/audit_combat_spatial_ppo_checkpoint_cuda.py','--roots',control,quality],out/'checkpoint-audit.log')
        sealed_update(control);sealed_update(quality)
        evaluation=out/'evaluation';evaluation.mkdir()
        compiler=series/'q2episode.exe'
        template=read(series/'round-1/capture/series/plan.json')
        families=[t['episode']['id'] for t in template['tasks']]
        plans,entries,conditions=[],[],None
        for name,update in [('parent',parent),('control',control),('after',quality),('rules',None)]:
            weights=None
            if update:
                weights=evaluation/(name+'-weights.json')
                shutil.copyfile(update/'weights.json',weights)
                assert not read(weights)['deterministic']
            for label,offset in ([('stochastic-a',20261011),('stochastic-b',20261012)] if weights else [('baseline',0)]):
                path=compile_plan(compiler,template['registry_path'],repo,weights,evaluation/(name+'-'+label),
                                  'validation',families,28)
                plan=read(path);plan['policy_sampling_seed_offset']=offset;save(path,plan)
                current={t['episode']['id']:(t['seeds'],t['instances']) for t in plan['tasks']}
                if conditions is None:conditions=current
                assert current==conditions
                run([compiler,'--verify-plan',path,'--root',repo],path.parent/'verify.log')
                plans.append(path)
                entries.append(dict(model=name,label=label,root=plan['output_root'],plan=str(path),
                    plan_sha256=sha(path),source_weights_sha256=sha(weights) if weights else None,
                    deterministic_weights_sha256=sha(weights) if weights else None,
                    deterministic=False if weights else True,policy_sampling_seed_offset=offset))
        save(evaluation/'protocol.json',dict(version='combat_sequence_retention_common_development_v1',
             evaluations=entries,families=families,episodes_per_model=80,total_episodes=560,slots=16,timescale=2,
             comparison_reference='rules-baseline',training_protocol_sha256=sha(out/'protocol.json'),
             scope='Parent, reproduced unregularized update and sequence-retained update x two RNG arms, plus rules.80 common scenes; reused development, no independent test or promotion.'))
        assert verify_sources(repo,reference)==source_binding
        save(out/'progress.json',dict(stage='native_evaluation',episodes=560,promotion=None))
        pool=collect_compressed(repo,plans,evaluation)
        finalize(repo,evaluation,pool)
        run([sys.executable,repo/'scripts/report_combat_adaptation_comparison.py','--root',evaluation],out/'comparison.log')
        save(out/'progress.json',dict(stage='complete',comparison_sha256=sha(evaluation/'adaptation-comparison.json'),
             evaluation_root=str(evaluation),promotion=None))
    except Exception as error:
        save(out/'progress.json',dict(stage='failed',error=str(error),promotion=None))
        raise


if __name__=='__main__':
    main()
