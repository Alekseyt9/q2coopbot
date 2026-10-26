"""Collect fresh source worlds then paired native teachers, without PPO fitting."""
import argparse,subprocess,sys
from pathlib import Path
from process_combat_architecture_pool import read,save,sha

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed-offset',type=int,default=320)
    ap.add_argument('--max-roots',type=int,choices=(16,32),default=16)
    a=ap.parse_args();assert a.seed_offset>=0
    repo=Path(__file__).resolve().parents[1];out=a.out.resolve();assert not out.exists();out.mkdir(parents=True)
    save(out/'protocol.json',dict(version='combat_native_teacher_expansion_v2',source_cases=32,seed_offset=a.seed_offset,
        inventory_offset_count=128,max_roots=a.max_roots,max_instances=16,timescale=2,
        source_model_sha256=sha(repo/'workspace/artifacts/combat-wide-gate-train-batch02-v3-20261026/fit/weights.json'),
        selection='Fresh source seeds excluded against physical plan history; inventory and geometry admission before outcomes. Earliest eligible frame>=110 per source map, all eligible maps up to declared cap. All branches required.',
        training_performed=False,promotion=False))
    def run(label,script,args):
        save(out/'progress.json',dict(stage=label))
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            p=subprocess.run([sys.executable,str(repo/'scripts'/script),*map(str,args)],cwd=repo,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert p.returncode==0,str(out/(label+'.log'))
    try:
        run('source32','run_combat_wide_gate_pilot.py',[
            '--source',repo/'workspace/artifacts/combat-wide-gate-train-batch02-v3-20261026/fit',
            '--config',repo/'workspace/artifacts/combat-wide-aim-gate-only-init-v1-20261026/enabled/config.json',
            '--reference-plan',repo/'workspace/artifacts/combat-wide-gate-pilot-v1-20261026/candidate-plan.json',
            '--compatibility',repo/'workspace/artifacts/combat-wide-aim-native-compat-v1-20261026/analysis-v2',
            '--cases',32,'--seed-offset',a.seed_offset,'--collect-only','--port',54500,'--out',out/'source32'])
        source=read(out/'source32/complete.json');assert source['collection_only'] and not source['training_performed'] and source['native_cases']==source['cuda_cases']==32
        run('paired','run_combat_wide_gate_teacher_pilot.py',['--collection',out/'source32','--max-roots',a.max_roots,'--out',out/'paired'])
        paired=read(out/'paired/complete.json');assert paired['diagnostic_only'] and not paired['training_performed']
        run('corpus','build_combat_native_teacher_registry.py',['--source',out/'paired','--out',out/'corpus'])
        corpus=read(out/'corpus/complete.json');manifest=read(out/'corpus/manifest.json')
        assert not corpus['ordinary_ppo_eligible'] and not corpus['training_performed'] and corpus['records']>=8
        save(out/'complete.json',dict(state='complete',source_native_cases=32,source_cuda_cases=32,paired_native_cases=paired['native_cases'],paired_cuda_cases=paired['cuda_cases'],teacher_records=corpus['records'],counts=manifest['counts'],
            source_sha256=sha(out/'source32/complete.json'),paired_sha256=sha(out/'paired/complete.json'),corpus_sha256=sha(out/'corpus/complete.json'),manifest_sha256=sha(out/'corpus/manifest.json'),
            training_performed=False,ordinary_ppo_eligible=False,promotion=False))
        save(out/'progress.json',dict(stage='complete'));print(out/'complete.json')
    except BaseException as e:save(out/'progress.json',dict(stage='failed',error=str(e)));raise

if __name__=='__main__':main()
