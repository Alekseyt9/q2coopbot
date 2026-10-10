"""Fresh CUDA PPO from two development-selected coarse/fine BC actors."""
import argparse,os,pathlib,shutil,sys
from process_combat_architecture_pool import read,save,sha,run
from run_combat_target_refresh import compile_plan,pool

def main():
    ap=argparse.ArgumentParser()
    for name in ('development','training','capture','processing'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--seed-offset',type=int,default=160)
    ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--run-prepared',action='store_true')
    a=ap.parse_args();assert not (a.prepare_only and a.run_prepared)
    repo=pathlib.Path(__file__).resolve().parents[1]
    dev,training,capture,processing=[p.resolve() for p in (a.development,a.training,a.capture,a.processing)]
    os.environ['GOCACHE']=str(repo/'workspace/build/go-cache');os.environ['GOTOOLCHAIN']='auto'
    quality=read(dev/'quality-report.json');proof=read(dev/'recovery/verified-members.json');protocol=read(dev/'protocol.json')
    assert quality['state']==proof['state']=='complete' and quality['episodes']==len(proof['members'])==960
    assert quality['protocol_sha256']==proof['protocol_sha256']==sha(dev/'protocol.json')
    assert not processing.exists()
    exporter=repo/'workspace/build/q2ppo-data-spatial-coarse-v2.exe'
    if not a.run_prepared:
        assert not capture.exists() and a.seed_offset>=144;capture.mkdir()
        report=read(training/'report.json');assert report['state']=='complete' and report['device']=='cuda'
        assert protocol['training_report_sha256']==sha(training/'report.json')
        variants=quality['variants'];best=min((v for v in variants if v['variant'].endswith('-after')),key=lambda v:(-v['wins'],v['deaths'],v['mean_received_damage'],v['variant']))
        selected=list(dict.fromkeys([best['variant'].removesuffix('-after'),'parent3']))
        assert len(selected)==2,'This paired experiment requires distinct best-win and retained PPO actors'
        template=read(protocol['evaluations'][0]['plan']);families=[t['episode']['id'] for t in template['tasks']]
        assert len(set(families))==20
        compiler=repo/'workspace/build/q2episode-spatial-v2.exe'
        run(['go','build','-buildvcs=false','-o',exporter,'./cmd/q2ppo-data'],capture/'build-exporter.log')
        entries={e['model']:e for e in report['models']};bindings=[];plans=[]
        for name in selected:
            source=training/name/'weights.json';entry=entries[name]
            assert entry['weights_sha256']==sha(source) and entry['checkpoint_sha256']==sha(training/name/'checkpoint.pt')
            assert entry['cuda_export_and_checkpoint_exact'] and entry['device']=='cuda'
            weights=read(source);assert weights['spatial_aim']['version']=='combat_shared_spatial_coarse_fine_aim_v2'
            weights.update(deterministic=False,sampling_seed=20261010)
            frozen=capture/(name+'-weights.json');save(frozen,weights)
            plan=compile_plan(compiler,template['registry_path'],repo,frozen,capture/name,'train',families,a.seed_offset)
            schedule=read(plan);assert all(t['split']=='train' for t in schedule['tasks'])
            seeds={s for t in schedule['tasks'] for s in t['seeds']};assert len(seeds)==80
            assert not seeds&{s for t in template['tasks'] for s in t['seeds']}
            bindings.append(dict(id=name,architecture=dict(id='coarse-fine-'+entry['architecture'],architecture=entry['architecture']),model=str(frozen),plan=str(plan),capture_root=schedule['output_root'],parent_sha256=sha(source),parent_report_sha256=sha(training/name/'report.json')))
            plans.append(str(plan))
        config=read(repo/'scripts/scenarios/combat-ppo-recoil-v5.json');config.update(actor_lr=.00003,target_kl=.005,seed=20261010)
        save(capture/'config.json',config);save(capture/'models.json',bindings);save(capture/'plans.json',plans)
        save(capture/'preparation.json',dict(state='prepared',episodes=160,slots=16,timescale=2,training_device='cuda',models_sha256=sha(capture/'models.json'),config_sha256=sha(capture/'config.json'),plans_sha256={p:sha(p) for p in plans},exporter_sha256=sha(exporter),development_quality_sha256=sha(dev/'quality-report.json'),training_report_sha256=sha(training/'report.json'),train_seed_offset=a.seed_offset,selected=selected,scope='Two development-selected actors,80 fresh own-policy fights each,common train conditions. New PPO Adam after BC changed the actor; inherited actor/value/std retained, no optimizer continuation claim. Whole actor including shared coarse/fine branch trained on CUDA, original reward objective, no final test or promotion.'))
    prep=read(capture/'preparation.json');assert prep['state']=='prepared'
    assert prep['development_quality_sha256']==sha(dev/'quality-report.json') and prep['training_report_sha256']==sha(training/'report.json')
    assert prep['models_sha256']==sha(capture/'models.json') and prep['config_sha256']==sha(capture/'config.json')
    assert prep['exporter_sha256']==sha(exporter)
    for path,digest in prep['plans_sha256'].items():assert sha(path)==digest
    if a.prepare_only:print('Prepared160 own-policy fights; not dispatched',flush=True);return
    assert read(dev/'progress.json')['diagnostics_complete'] and not (capture/'pool').exists()
    try:
        save(capture/'execution.json',dict(stage='collecting_own_policy',episodes=160))
        pool(repo,repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe',[pathlib.Path(p) for p in read(capture/'plans.json')],capture/'pool',capture/'collect.log')
        save(capture/'execution.json',dict(stage='cuda_processing',episodes=160))
        legacy=repo/'workspace/artifacts/aproc-v1-20261007'
        run([sys.executable,repo/'scripts/process_combat_architecture_pool.py','--capture-root',capture,'--out',processing,'--config',capture/'config.json','--exporter',exporter,'--anchor',legacy/'anchor.json','--bank',legacy/'bank.json','--export-workers',4,'--cuda-only-export','--cuda-batch-finalize'],capture/'process.log')
        result=read(processing/'report.json');assert result['state']=='complete' and len(result['training'])==2
        assert all(e['device']=='cuda' and e['updates_completed']==1 for e in result['training'])
        run([sys.executable,repo/'scripts/audit_combat_spatial_ppo_checkpoint_cuda.py','--roots',*[processing/name/'update' for name in prep['selected']]],capture/'checkpoint-audit.log')
        save(capture/'execution.json',dict(stage='complete',episodes=160,processing_report_sha256=sha(processing/'report.json'),promotion=None))
    except Exception as error:
        save(capture/'execution.json',dict(stage='failed',error=str(error),promotion=None));raise

if __name__=='__main__':main()
