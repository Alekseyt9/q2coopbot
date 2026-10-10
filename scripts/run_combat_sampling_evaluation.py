"""Source-bound paired inference evaluation with explicit policy RNG offsets."""
import argparse, concurrent.futures, os, pathlib, shutil, subprocess, sys, time
from process_combat_architecture_pool import read, save, sha, run
from run_combat_target_refresh import compile_plan, pool
from run_combat_spatial_ppo import wait_process


def collect_compressed(repo, plans, out):
    """Compact only terminal successful member streams; verify byte preservation."""
    receipts=[]; seen=set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future=executor.submit(pool,repo,repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe',plans,out/'pool',out/'evaluation.log')
        while True:
            for path in (out/'pool/jobs').glob('*-result.json'):
                if path.name in seen: continue
                try: job=read(path)
                except (OSError,ValueError): continue  # publisher may still be writing
                if job['error']: seen.add(path.name); continue
                folder=pathlib.Path(job['root']).resolve()
                assert folder.is_relative_to(out)
                report=read(folder/'report.json')
                assert report['capture_complete'] and report['provenance_valid']
                for stream in folder.rglob('*'):
                    if not stream.is_file() or stream.suffix not in ('.jsonl','.log','.json') or stream.stat().st_size<131072: continue
                    digest=sha(stream)
                    result=subprocess.run(['compact.exe','/C','/EXE:LZX','/F','/Q',str(stream)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
                    assert result.returncode==0, result.stderr.decode(errors='replace')
                    assert sha(stream)==digest, 'Stream bytes changed during LZX compression'
                    receipts.append(dict(path=str(stream),sha256=digest,logical_bytes=stream.stat().st_size))
                seen.add(path.name)
                save(out/'stream-compression-receipts.json',dict(state='running',completed_members=len(seen),files=receipts))
            if future.done():
                result=future.result()
                if len(seen)==len(result['jobs']): break
            time.sleep(2)
    save(out/'stream-compression-receipts.json',dict(state='complete',completed_members=len(seen),files=receipts,
        scope='Terminal successful members only; native LZX compression with exact SHA256 preservation per stream. No logical path changes.'))
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--reference',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True); ap.add_argument('--wide',action='store_true')
    ap.add_argument('--require-smoke',type=pathlib.Path); ap.add_argument('--wait-pid',type=int)
    a=ap.parse_args(); repo=pathlib.Path(__file__).resolve().parents[1]; out=a.out.resolve(); reference=a.reference.resolve()
    assert not out.exists(); out.mkdir()
    os.environ['GOCACHE']=str(repo/'workspace/build/go-cache'); os.environ['GOTOOLCHAIN']='auto'
    try:
        if a.wait_pid: wait_process(a.wait_pid,out/'progress.json','waiting_for_sampling_smoke')
        if a.wide:
            assert a.require_smoke is not None
            smoke=a.require_smoke.resolve(); assert read(smoke/'progress.json')['diagnostics_complete']
            audit=read(smoke/'sampling-config-audit.json')
            assert audit['state']=='complete' and not audit['repeated_stochastic_execution_configs']
            assert audit['protocol_sha256']==sha(smoke/'protocol.json')
            ownership=read(smoke/'control-ownership.json')
            assert ownership['state']=='complete' and ownership['protocol_sha256']==sha(smoke/'protocol.json')
            assert ownership['quality_sha256']==sha(smoke/'quality-report.json')
            for name,group in ownership['groups'].items():
                if name=='rules-baseline': continue
                assert not group['counts'].get('rules_with_clear_target',0), 'Visible-target rules fallback must be resolved before wider comparison'
                assert not group['fallback_reasons'].get('pilot_equip_not_ready',0), 'Supported pickup ownership not resolved'
            smokeproof=read(smoke/'recovery/verified-members.json')
            for field,base in [('sources',repo),('native_sources',repo.parent/'yquake2')]:
                first=next(iter(smokeproof['members'])); manifest=read(pathlib.Path(first)/'manifest.json')
                for record in manifest[field]: assert sha(base/record['path'])==record['sha256']
        reserve_gib=10 if a.wide else 2
        assert shutil.disk_usage(out).free>reserve_gib*1024**3, f'Reserve at least{reserve_gib}GiB before collection'
        assert read(reference/'progress.json')['diagnostics_complete']
        old=read(reference/'protocol.json'); oldquality=read(reference/'quality-report.json')
        assert oldquality['protocol_sha256']==sha(reference/'protocol.json')
        families=old['families'] if a.wide else ['campaign-base2-site-02-blaster','campaign-base2-site-02-machinegun']
        plans=[]; entries=[]
        for entry in old['evaluations']:
            if entry['label']!='after' and entry['model']!='rules': continue
            original_plan=read(entry['plan'])
            assert sha(entry['plan'])==entry['plan_sha256']
            source=pathlib.Path(original_plan['model_path']) if entry['model']!='rules' else None
            if source: assert sha(source)==entry['deterministic_weights_sha256']
            modes=[('deterministic',True,0),('stochastic-a',False,20261011),('stochastic-b',False,20261012)] if source else [('baseline',True,0)]
            for label,deterministic,offset in modes:
                weights=None
                if source:
                    model=read(source); model['deterministic']=deterministic
                    weights=out/(entry['model']+'-'+label+'-weights.json'); save(weights,model)
                plan=compile_plan(repo/'workspace/build/q2episode-sampling-v1.exe',original_plan['registry_path'],repo,weights,
                    out/(entry['model']+'-'+label),'validation',families,24)
                compiled=read(plan); compiled['policy_sampling_seed_offset']=offset; save(plan,compiled)
                wanted=[t for t in original_plan['tasks'] if t['episode']['id'] in families]
                assert {t['episode']['id']:(t['seeds'],t.get('instances')) for t in compiled['tasks']}=={t['episode']['id']:(t['seeds'],t.get('instances')) for t in wanted}
                plans.append(plan); entries.append(dict(model=entry['model'],label=label,root=compiled['output_root'],plan=str(plan),plan_sha256=sha(plan),
                    source_weights_sha256=sha(source) if source else None,deterministic_weights_sha256=sha(weights) if weights else None,
                    deterministic=deterministic,policy_sampling_seed_offset=offset))
        count=len(families)*4; total=count*len(entries); assert len(entries)==7
        save(out/'protocol.json',dict(version='combat_sampling_offset_development_v1',evaluations=entries,families=families,
            episodes_per_model=count,total_episodes=total,slots=16,timescale=2,comparison_reference='rules-baseline',
            reference_protocol_sha256=sha(reference/'protocol.json'),smoke_protocol_sha256=sha(a.require_smoke/'protocol.json') if a.require_smoke else None,
            scope='Paired deterministic and two stochastic execution modes of identical updated actors. Effective policy RNG seed=engine seed+declared offset. Reused development24..27; no training or independent test. Whole-action sampling, not isolated fire intervention.'))
        save(out/'progress.json',dict(stage='evaluating',episodes=total,promotion=None))
        result=collect_compressed(repo,plans,out)
        manifest=read(pathlib.Path(result['jobs'][0]['root'])/'manifest.json')
        run([sys.executable,repo/'scripts/verify_combat_evaluation_members.py','--root',out,'--source-fingerprint',manifest['source_fingerprint'],
            '--native-fingerprint',manifest['native_source_fingerprint']],out/'verify.log')
        run([sys.executable,repo/'scripts/audit_combat_sampling_configs.py','--root',out],out/'sampling-config-audit.log')
        audit=read(out/'sampling-config-audit.json'); assert not audit['repeated_stochastic_execution_configs']
        run([sys.executable,repo/'scripts/report_combat_architecture_evaluation.py','--root',out,'--member-proof',out/'recovery/verified-members.json'],out/'quality.log')
        run([sys.executable,repo/'scripts/finalize_combat_machinegun_evaluation.py','--root',out],out/'diagnostics.log')
        run([sys.executable,repo/'scripts/report_combat_visible_target_decisions.py','--root',out,'--families',*families,'--out',out/'visible-target-decisions.json'],out/'visible-target-decisions.log')
        run([sys.executable,repo/'scripts/report_combat_control_ownership.py','--root',out],out/'control-ownership.log')
        ownership=read(out/'control-ownership.json')
        progress=read(out/'progress.json'); progress.update(control_ownership_complete=True,control_ownership_sha256=sha(out/'control-ownership.json'),
            learned_visible_target_rules_fallback={name:group['counts'].get('rules_with_clear_target',0) for name,group in ownership['groups'].items() if name!='rules-baseline'})
        save(out/'progress.json',progress)
        files={name:[p for p in out.glob('*/capture/case-*/s-*/'+name)] for name in ('q2coopbot.exe','q2combat-export.exe')}
        save(out/'binary-storage-audit.json',{name:dict(paths=len(paths),unique_inodes=len({(p.stat().st_dev,p.stat().st_ino) for p in paths})) for name,paths in files.items()})
        print(f'Sealed {total} paired development episodes with distinct policy RNG configs',flush=True)
    except Exception as error:
        save(out/'progress.json',dict(stage='failed',error=str(error),promotion=None)); raise


if __name__=='__main__': main()
