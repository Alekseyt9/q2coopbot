"""Paired inference-mode experiment only if development still trails rules."""
import argparse,pathlib,sys
from process_combat_architecture_pool import read,save,sha,run
from run_combat_target_refresh import compile_plan,pool
from run_combat_spatial_ppo import wait_process

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evaluation',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--wait-pid',type=int);a=ap.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[1];evaluation=a.evaluation.resolve();out=a.out.resolve()
    assert not out.exists();out.mkdir()
    try:
        if a.wait_pid:wait_process(a.wait_pid,out/'progress.json','waiting_for_paired_development')
        progress=read(evaluation/'progress.json');assert progress['stage']=='complete' and progress['diagnostics_complete']
        quality=read(evaluation/'quality-report.json');protocol=read(evaluation/'protocol.json');proof=read(evaluation/'recovery/verified-members.json')
        assert quality['state']==proof['state']=='complete' and quality['protocol_sha256']==proof['protocol_sha256']==sha(evaluation/'protocol.json')
        variants={v['variant']:v for v in quality['variants']};rules=variants['rules-baseline'];afters=[v for k,v in variants.items() if k.endswith('-after')]
        best=min(afters,key=lambda v:(-v['wins'],v['deaths'],v['mean_received_damage'],v['variant']))
        eligible=best['wins']>rules['wins'] or best['wins']==rules['wins'] and best['deaths']<rules['deaths']
        save(out/'decision.json',dict(development_quality_sha256=sha(evaluation/'quality-report.json'),best=best['variant'],best_wins=best['wins'],rules_wins=rules['wins'],independent_gate_passed=eligible))
        if eligible:
            save(out/'progress.json',dict(stage='deferred_independent_gate_passed',reason='Prioritize predeclared independent test; no sampling pilot dispatched'));return
        families=['campaign-base2-site-02-blaster','campaign-base2-site-02-machinegun']
        if all(sum(v['families'][f]['wins'] for f in families)>=sum(rules['families'][f]['wins'] for f in families) for v in afters):
            save(out/'progress.json',dict(stage='site02_gap_closed',reason='No remaining site02 win gap; no sampling pilot dispatched'));return
        entries=[];plans=[]
        for entry in protocol['evaluations']:
            if entry['label']!='after' and entry['model']!='rules':continue
            sourceplan=read(entry['plan']);source=pathlib.Path(sourceplan['model_path']) if entry['model']!='rules' else None
            if source:assert sha(source)==entry['deterministic_weights_sha256']
            modes=[('deterministic',True,20261010),('stochastic-a',False,20261011),('stochastic-b',False,20261012)] if source else [('baseline',True,None)]
            for label,deterministic,sampling_seed in modes:
                name=entry['model'];weights=None
                if source:
                    original=read(source);model=dict(original,deterministic=deterministic,sampling_seed=sampling_seed)
                    assert {k:v for k,v in model.items() if k not in ('deterministic','sampling_seed')}=={k:v for k,v in original.items() if k not in ('deterministic','sampling_seed')}
                    weights=out/(name+'-'+label+'-weights.json');save(weights,model)
                plan=compile_plan(repo/'workspace/build/q2episode-spatial-v2.exe',sourceplan['registry_path'],repo,weights,out/(name+'-'+label),'validation',families,24)
                compiled=read(plan);wanted=[t for t in sourceplan['tasks'] if t['episode']['id'] in families]
                assert {t['episode']['id']:t['seeds'] for t in compiled['tasks']}=={t['episode']['id']:t['seeds'] for t in wanted}
                assert {t['episode']['id']:t['episode'] for t in compiled['tasks']}=={t['episode']['id']:t['episode'] for t in wanted}
                plans.append(plan);entries.append(dict(model=name,label=label,root=str(out/(name+'-'+label)/'capture'),plan=str(plan),plan_sha256=sha(plan),source_weights_sha256=sha(source) if source else None,deterministic_weights_sha256=sha(weights) if weights else None,deterministic=deterministic,sampling_seed=sampling_seed))
        assert len(entries)==7
        save(out/'protocol.json',dict(version='combat_action_sampling_site02_pilot_v1',evaluations=entries,families=families,episodes_per_model=8,total_episodes=56,slots=16,timescale=2,comparison_reference='rules-baseline',development_quality_sha256=sha(evaluation/'quality-report.json'),scope='Same updated actors and registered development conditions; deterministic versus two fixed RNG seeds sampling the entire existing policy action distribution. No parameters changed, no training or demonstrations. Does not isolate fire from movement/vertical/weapon randomness. Small development pilot only, no independent-test or general superiority claim.'))
        save(out/'progress.json',dict(stage='evaluating',episodes=56))
        pool(repo,repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe',plans,out/'pool',out/'evaluation.log')
        manifest=read(next(pathlib.Path(entries[0]['root']).glob('case-*/s-*/manifest.json')))
        run([sys.executable,repo/'scripts/verify_combat_evaluation_members.py','--root',out,'--source-fingerprint',manifest['source_fingerprint'],'--native-fingerprint',manifest['native_source_fingerprint']],out/'verify.log')
        run([sys.executable,repo/'scripts/report_combat_architecture_evaluation.py','--root',out,'--member-proof',out/'recovery/verified-members.json'],out/'quality.log')
        save(out/'progress.json',dict(stage='quality_complete',episodes=56,quality_report_sha256=sha(out/'quality-report.json'),promotion=None))
        run([sys.executable,repo/'scripts/finalize_combat_machinegun_evaluation.py','--root',out],out/'diagnostics.log')
        run([sys.executable,repo/'scripts/report_combat_visible_target_decisions.py','--root',out,'--families',*families,'--out',out/'visible-target-decisions.json'],out/'visible-target-decisions.log')
    except Exception as error:
        save(out/'progress.json',dict(stage='failed',error=str(error),promotion=None));raise

if __name__=='__main__':main()
