"""Continue both sealed CUDA optimizers only after clean pickup ownership proof."""
import argparse,os,pathlib,shutil,sys
from process_combat_architecture_pool import read,save,sha,run
from run_combat_target_refresh import compile_plan
from run_combat_sampling_evaluation import collect_compressed
from run_combat_spatial_ppo import wait_process


def main():
    ap=argparse.ArgumentParser()
    for name in ('parent','reference','fix','capture','processing'): ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--seed-offset',type=int,default=164); ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--run-prepared',action='store_true'); ap.add_argument('--wait-pid',type=int)
    a=ap.parse_args(); assert not(a.prepare_only and a.run_prepared)
    repo=pathlib.Path(__file__).resolve().parents[1]
    parent,reference,fix,capture,processing=[p.resolve() for p in (a.parent,a.reference,a.fix,a.capture,a.processing)]
    assert not processing.exists()
    os.environ['GOCACHE']=str(repo/'workspace/build/go-cache'); os.environ['GOTOOLCHAIN']='auto'
    if not a.run_prepared:
        assert not capture.exists() and a.seed_offset>=164; capture.mkdir()
        report=read(parent/'report.json'); assert report['state']=='complete' and len(report['training'])==2
        protocol=read(reference/'protocol.json'); template=read(protocol['evaluations'][0]['plan'])
        families=protocol['families']; assert len(families)==20
        # Exact bytes preserve the parent optimizer contract, including config RNG.
        shutil.copyfile(parent/'config.json',capture/'config.json')
        bindings=[]; plans=[]; entries=[]
        for prior in report['training']:
            name=prior['model']; update=parent/name/'update'; seal=read(update/'complete.json'); audit=read(update/'checkpoint-cuda-audit.json'); prior_report=read(update/'report.json')
            assert prior['device']=='cuda' and audit['device']=='cuda' and audit['actor_value_std_exact'] and audit['optimizer_state_exact']
            for file,field in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]: assert sha(update/file)==seal[field]
            assert audit['weights_sha256']==sha(update/'weights.json') and audit['checkpoint_sha256']==sha(update/'checkpoint.pt')
            assert prior_report['config_sha256']==sha(capture/'config.json')
            for filename,field in [('anchor.json','anchor_sha256'),('bank.json','bank_sha256')]: assert sha(parent/filename)==prior_report[field]
            weights=capture/(name+'-weights.json'); shutil.copyfile(update/'weights.json',weights)
            assert not read(weights)['deterministic'] and sha(weights)==prior_report['weights_sha256']
            plan=compile_plan(repo/'workspace/build/q2episode-sampling-v1.exe',template['registry_path'],repo,weights,capture/name,'train',families,a.seed_offset)
            schedule=read(plan); assert all(t['split']=='train' for t in schedule['tasks'])
            assert len({seed for t in schedule['tasks'] for seed in t['seeds']})==80
            bindings.append(dict(id=name,architecture=prior['architecture'],model=str(weights),plan=str(plan),capture_root=schedule['output_root'],
                resume_checkpoint=str(update/'checkpoint.pt'),resume_checkpoint_sha256=sha(update/'checkpoint.pt'),
                resume_report=str(update/'report.json'),resume_report_sha256=sha(update/'report.json'),parent_updates_completed=prior_report['updates_completed']))
            plans.append(str(plan)); entries.append(dict(model=name,label='behavior',root=schedule['output_root'],plan=str(plan),plan_sha256=sha(plan),
                deterministic_weights_sha256=sha(weights),source_weights_sha256=sha(update/'weights.json')))
        save(capture/'models.json',bindings); save(capture/'plans.json',plans)
        save(capture/'protocol.json',dict(version='combat_pickup_clean_onpolicy_v1',evaluations=entries,families=families,episodes_per_model=80,total_episodes=160,
            comparison_reference='parent3-behavior',scope='Training outcomes/ownership only.160 fresh native on-policy captures, train164..167, same registered20 families. Not unseen-history or validation/test claims. Exact CUDA actor/value/std/Adam continuation, unchanged config/reward.'))
        save(capture/'preparation.json',dict(state='prepared',models_sha256=sha(capture/'models.json'),config_sha256=sha(capture/'config.json'),
            plans_sha256={p:sha(p) for p in plans},parent_processing_sha256=sha(parent/'report.json'),protocol_sha256=sha(capture/'protocol.json'),
            slots=16,timescale=2,training_device='cuda',episodes=160,seed_offset=a.seed_offset,required_fix=str(fix),promotion=None))
    prep=read(capture/'preparation.json')
    assert prep['models_sha256']==sha(capture/'models.json') and prep['config_sha256']==sha(capture/'config.json')
    assert prep['protocol_sha256']==sha(capture/'protocol.json') and prep['parent_processing_sha256']==sha(parent/'report.json')
    for path,digest in prep['plans_sha256'].items(): assert sha(path)==digest
    if a.prepare_only: print('Prepared160 own-policy episodes and exact CUDA optimizer continuation; not dispatched',flush=True); return
    try:
        if a.wait_pid: wait_process(a.wait_pid,capture/'execution.json','waiting_for_clean_pickup_validation')
        fixed=read(fix/'progress.json'); assert fixed['stage']=='complete' and fixed['rule_attack_with_clear_target']==0
        live=fix/'live-validation'; ownership=read(live/'control-ownership.json')
        assert fixed['control_ownership_sha256']==sha(live/'control-ownership.json')
        assert ownership['quality_sha256']==fixed['live_quality_sha256']==sha(live/'quality-report.json')
        for name,group in ownership['groups'].items():
            if name=='rules-baseline': continue
            assert not group['counts'].get('rules_with_clear_target',0) and not group['fallback_reasons'].get('pilot_equip_not_ready',0)
        assert shutil.disk_usage(capture).free>4*1024**3, 'Reserve4GiB before160-case capture and processing'
        assert not(capture/'pool').exists()
        save(capture/'execution.json',dict(stage='collecting_own_policy',episodes=160,parent_optimizer_preserved=True))
        result=collect_compressed(repo,[pathlib.Path(p) for p in read(capture/'plans.json')],capture)
        manifest=read(pathlib.Path(result['jobs'][0]['root'])/'manifest.json')
        run([sys.executable,repo/'scripts/verify_combat_evaluation_members.py','--root',capture,'--source-fingerprint',manifest['source_fingerprint'],
            '--native-fingerprint',manifest['native_source_fingerprint']],capture/'verify.log')
        run([sys.executable,repo/'scripts/report_combat_architecture_evaluation.py','--root',capture,'--member-proof',capture/'recovery/verified-members.json'],capture/'training-outcomes.log')
        run([sys.executable,repo/'scripts/report_combat_control_ownership.py','--root',capture],capture/'ownership.log')
        own=read(capture/'control-ownership.json'); assert all(not v['counts'].get('rules_with_clear_target',0) for v in own['groups'].values())
        assert all(not v['fallback_reasons'].get('pilot_equip_not_ready',0) for v in own['groups'].values())
        save(capture/'execution.json',dict(stage='cuda_processing',episodes=160,control_ownership_sha256=sha(capture/'control-ownership.json')))
        exporter=parent/'q2ppo-data.exe'
        run([sys.executable,repo/'scripts/process_combat_architecture_pool.py','--capture-root',capture,'--out',processing,'--config',capture/'config.json',
            '--exporter',exporter,'--anchor',parent/'anchor.json','--bank',parent/'bank.json','--export-workers',4,'--cuda-only-export','--cuda-batch-finalize'],capture/'process.log')
        result=read(processing/'report.json'); assert result['state']=='complete' and len(result['training'])==2
        bindings={b['id']:b for b in read(capture/'models.json')}
        for entry in result['training']:
            binding=bindings[entry['model']]
            assert entry['device']=='cuda' and entry['updates_completed']==binding['parent_updates_completed']+1
            assert entry['resume_checkpoint_sha256']==binding['resume_checkpoint_sha256']
        run([sys.executable,repo/'scripts/audit_combat_spatial_ppo_checkpoint_cuda.py','--roots',*[processing/name/'update' for name in bindings]],capture/'checkpoint-audit.log')
        save(capture/'execution.json',dict(stage='complete',episodes=160,processing_report_sha256=sha(processing/'report.json'),parent_optimizer_preserved=True,promotion=None))
    except Exception as error:
        save(capture/'execution.json',dict(stage='failed',error=str(error),promotion=None)); raise


if __name__=='__main__': main()
