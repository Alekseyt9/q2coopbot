"""Four fresh native registry rounds, CUDA checkpoint continuation, paired eval."""
import argparse,json,pathlib,shutil,sys
from process_combat_architecture_pool import read,sha,save,run


def main():
    ap=argparse.ArgumentParser()
    for name in ('parent','compiler','teacher','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--models',required=True);a=ap.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[1];out=a.out.resolve();parent=a.parent.resolve()
    assert not out.exists();out.mkdir();shutil.copy2(a.compiler,out/'q2episode.exe')
    source={p.name:sha(p) for p in (repo/'scripts').glob('*.py')}
    protocol=dict(version='combat_checkpoint_series_v1',initial_parent=str(parent),initial_report_sha256=sha(parent/'report.json'),
        teacher=str(a.teacher.resolve()),teacher_sha256=sha(a.teacher),models=a.models.split(','),round_offsets=[72,80,88,96],
        episodes_per_model_per_round=80,total_training_episodes=640,slots=16,timescale=2,device='cuda',python_sources=source)
    assert len(protocol['models'])==2
    save(out/'protocol.json',protocol);rounds=[]
    pwsh=repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe'
    try:
        for i,offset in enumerate(protocol['round_offsets'],1):
            assert all(sha(repo/'scripts'/name)==digest for name,digest in source.items()),'Frozen Python sources changed'
            root=out/f'round-{i}';root.mkdir();capture=root/'capture';processing=root/'processing'
            save(out/'progress.json',dict(stage='preparing',round=i,seed_offset=offset))
            run([sys.executable,repo/'scripts/prepare_combat_checkpoint_training.py','--parent',parent,
                 '--compiler',out/'q2episode.exe','--out',capture,'--models',a.models,'--seed-offset',offset],root/'prepare.log')
            run([sys.executable,repo/'scripts/verify_combat_checkpoint_cuda.py','--models',capture/'models.json',
                 '--out',capture/'cuda-checkpoint-proof.json'],root/'cuda-proof.log')
            save(out/'progress.json',dict(stage='native_capture_then_cuda',round=i,seed_offset=offset,capture_root=str(capture)))
            run([pwsh,'-NoProfile','-File',repo/'scripts/run_combat_distillation_training.ps1',
                 '-CaptureRoot',capture,'-ProcessingRoot',processing,'-Port',34600],root/'driver.log')
            report=read(processing/'report.json');assert report['state']=='complete' and len(report['training'])==2
            assert all(u['updates_completed']==i+1 and u['device']=='cuda' and u['allocated_episodes']==80 for u in report['training'])
            rounds.append(dict(round=i,seed_offset=offset,processing_root=str(processing),report_sha256=sha(processing/'report.json'),training=report['training']))
            save(out/'rounds.json',rounds);parent=processing
        evaluation=out/'evaluation';save(out/'progress.json',dict(stage='preparing_evaluation'))
        assert sha(a.teacher)==protocol['teacher_sha256']
        run([sys.executable,repo/'scripts/prepare_combat_architecture_evaluation.py','--processing-root',parent,
             '--before-processing-root',a.parent.resolve(),'--out',evaluation,'--compiler',out/'q2episode.exe',
             '--teacher',a.teacher.resolve(),'--seed-offset',24,'--count',4],out/'prepare-evaluation.log')
        assert read(evaluation/'protocol.json')['total_episodes']==480
        save(out/'progress.json',dict(stage='evaluating',episodes=480))
        run([pwsh,'-NoProfile','-File',repo/'scripts/run_combat_architecture_evaluation.ps1',
             '-EvaluationRoot',evaluation,'-Port',34700],out/'evaluation.log')
        quality=read(evaluation/'quality-report.json');assert quality['state']=='complete' and quality['episodes']==480
        save(out/'report.json',dict(state='complete',rounds=rounds,quality_report_sha256=sha(evaluation/'quality-report.json'),
                                   promotion='Requires review; reused validation, not final-test superiority'))
        save(out/'progress.json',dict(stage='complete'))
    except Exception as error:
        save(out/'report.json',dict(state='failed',error=str(error),rounds=rounds));raise


if __name__=='__main__':main()
