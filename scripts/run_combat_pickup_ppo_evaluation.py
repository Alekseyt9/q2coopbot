"""Paired original/continued actors under clean pickup ownership, all modes."""
import argparse,pathlib,shutil,sys
from process_combat_architecture_pool import read,save,sha,run
from run_combat_target_refresh import compile_plan
from run_combat_sampling_evaluation import collect_compressed
from run_combat_spatial_ppo import wait_process


def main():
    ap=argparse.ArgumentParser()
    for name in ('capture','processing','reference','out'): ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--wait-pid',type=int); a=ap.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[1]; capture=a.capture.resolve(); processing=a.processing.resolve(); reference=a.reference.resolve(); out=a.out.resolve()
    assert not out.exists(); out.mkdir()
    # Aggregate exporter copies must use the same stable bundle shard pool.
    pool_script=repo/'scripts/run_registered_combat_episode_pool.ps1'; pool_hash=sha(pool_script)
    before="        Copy-Item -LiteralPath $groupExporter -Destination (Join-Path $g.root 'q2combat-export.exe')"
    after="        $aggregateSource=if($binaryBundle){Join-Path $binaryBundle 'q2combat-export.exe'}else{$groupExporter}\n        Install-HarnessBinaryLink $aggregateSource (Join-Path $g.root 'q2combat-export.exe') $manifest.exporter_sha256.ToLowerInvariant()"
    original=pool_script.read_text(encoding='utf-8'); assert original.count(before)==1
    save(out/'preparation.json',dict(pool_script_before_sha256=pool_hash,planned_episodes=1040,
        scope='Two models before/after resumed CUDA PPO, three inference modes,80 common development conditions per arm plus rules. No test or promotion. Aggregate exporters linked from stable bundle, large terminal JSON configs losslessly LZX-compressed.'))
    try:
        if a.wait_pid: wait_process(a.wait_pid,out/'progress.json','waiting_for_cuda_continuation')
        execution=read(capture/'execution.json'); assert execution['stage']=='complete' and execution['parent_optimizer_preserved']
        assert execution['processing_report_sha256']==sha(processing/'report.json')
        assert sha(pool_script)==pool_hash, 'Pool source changed while queued'
        (out/'pool-script-before.ps1.txt').write_bytes(pool_script.read_bytes())
        pool_script.write_text(original.replace(before,after),encoding='utf-8',newline='\n')
        save(out/'pool-source-change.json',dict(before_sha256=pool_hash,after_sha256=sha(pool_script),
            scope='Only aggregate exporter Copy-Item replaced with verified shared binary installer; stable bundle source preserves NTFS shard reuse. Applied after training capture and CUDA processing closed.'))
        assert shutil.disk_usage(out).free>10*1024**3, 'Reserve10GiB before1040-case evaluation'
        protocol=read(reference/'protocol.json'); template=read(protocol['evaluations'][0]['plan']); families=protocol['families']; assert len(families)==20
        assert read(reference/'progress.json')['diagnostics_complete']
        bindings=read(capture/'models.json'); plans=[]; entries=[]
        variants=[]
        for binding in bindings:
            name=binding['id']; update=processing/name/'update'; audit=read(update/'checkpoint-cuda-audit.json'); seal=read(update/'complete.json')
            assert audit['device']=='cuda' and audit['actor_value_std_exact'] and audit['optimizer_state_exact']
            for file,field in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]: assert sha(update/file)==seal[field]
            assert audit['weights_sha256']==sha(update/'weights.json') and audit['checkpoint_sha256']==sha(update/'checkpoint.pt')
            variants.extend([(name,'before',pathlib.Path(binding['model'])),(name,'after',update/'weights.json')])
        variants.append(('rules','baseline',None))
        for name,stage,source in variants:
            modes=[('deterministic',True,0),('stochastic-a',False,20261011),('stochastic-b',False,20261012)] if source else [('baseline',True,0)]
            for mode,deterministic,offset in modes:
                model_name=name+'-'+mode if source else name; label=stage if source else 'baseline'; weights=None
                if source:
                    model=read(source); model['deterministic']=deterministic
                    weights=out/(model_name+'-'+label+'-weights.json'); save(weights,model)
                plan=compile_plan(repo/'workspace/build/q2episode-sampling-v1.exe',template['registry_path'],repo,weights,out/(model_name+'-'+label),'validation',families,24)
                compiled=read(plan); compiled['policy_sampling_seed_offset']=offset; save(plan,compiled)
                assert {t['episode']['id']:(t['seeds'],t.get('instances')) for t in compiled['tasks']}=={t['episode']['id']:(t['seeds'],t.get('instances')) for t in template['tasks']}
                plans.append(plan); entries.append(dict(model=model_name,label=label,root=compiled['output_root'],plan=str(plan),plan_sha256=sha(plan),
                    deterministic_weights_sha256=sha(weights) if weights else None,source_weights_sha256=sha(source) if source else None,
                    policy_sampling_seed_offset=offset,deterministic=deterministic))
        assert len(entries)==13
        save(out/'protocol.json',dict(version='combat_pickup_ppo_paired_development_v1',evaluations=entries,families=families,episodes_per_model=80,total_episodes=1040,
            slots=16,timescale=2,comparison_reference='rules-baseline',processing_sha256=sha(processing/'report.json'),
            scope='Both original and continued actors executed under identical corrected pickup runtime; deterministic/two declared stochastic RNG offsets, reused development24..27. No new training in this evaluation, no independent test or promotion. Whole-action sampling.'))
        save(out/'progress.json',dict(stage='evaluating',episodes=1040,promotion=None))
        result=collect_compressed(repo,plans,out); manifest=read(pathlib.Path(result['jobs'][0]['root'])/'manifest.json')
        run([sys.executable,repo/'scripts/verify_combat_evaluation_members.py','--root',out,'--source-fingerprint',manifest['source_fingerprint'],
            '--native-fingerprint',manifest['native_source_fingerprint']],out/'verify.log')
        run([sys.executable,repo/'scripts/audit_combat_sampling_configs.py','--root',out],out/'sampling-config-audit.log')
        # Before/after may share identical weights if optimizer rejects every actor
        # step; offsets still checked against actual provider configs individually.
        run([sys.executable,repo/'scripts/report_combat_architecture_evaluation.py','--root',out,'--member-proof',out/'recovery/verified-members.json'],out/'quality.log')
        run([sys.executable,repo/'scripts/finalize_combat_machinegun_evaluation.py','--root',out],out/'diagnostics.log')
        run([sys.executable,repo/'scripts/report_combat_control_ownership.py','--root',out],out/'control-ownership.log')
        run([sys.executable,repo/'scripts/report_combat_visible_target_decisions.py','--root',out,'--families',*families,'--out',out/'visible-target-decisions.json'],out/'visible-target-decisions.log')
        own=read(out/'control-ownership.json'); learned={k:v for k,v in own['groups'].items() if k!='rules-baseline'}
        save(out/'ownership-acceptance.json',dict(state='complete',control_ownership_sha256=sha(out/'control-ownership.json'),
            visible_target_rules_frames={k:v['counts'].get('rules_with_clear_target',0) for k,v in learned.items()},
            equip_fallback_frames={k:v['fallback_reasons'].get('pilot_equip_not_ready',0) for k,v in learned.items()},promotion=None))
        run([sys.executable,repo/'scripts/audit_combat_physical_storage.py','--root',out,'--out',out/'storage-final.json'],out/'storage-final.log')
        progress=read(out/'progress.json'); progress.update(ownership_acceptance_sha256=sha(out/'ownership-acceptance.json'),physical_storage_sha256=sha(out/'storage-final.json'))
        save(out/'progress.json',progress)
    except Exception as error:
        save(out/'progress.json',dict(stage='failed',error=str(error),promotion=None)); raise


if __name__=='__main__': main()
