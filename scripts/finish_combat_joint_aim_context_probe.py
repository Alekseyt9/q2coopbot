"""Finalize a closed TRAIN diagnostic pool; never create quality evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from evaluate_registered_combat_models import read, save, sha
from compare_registered_combat_pools import runtime_bindings


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    root=a.root.resolve();repo=Path(__file__).resolve().parents[1];assert not (root/'complete.json').exists()
    pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    spec=read(pin(root/'spec.json'));assert spec['version']=='combat_joint_aim_context_probe_v1' and spec['split']=='train'
    assert spec['seed_reuse_intentional'] and not spec['training_performed'] and not spec['promotion']
    selected_count=len(spec['selected']);assert selected_count in (1,2)
    case_count=2*selected_count
    pool=read(pin(root/'pool/report.json'));assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==len(pool['jobs'])==case_count
    assert [v['label'] for v in spec['variants']]==['source','marginal']
    for p,h in spec['inputs'].items():pin(p,h)
    plans=[read(pin(b['path'],b['sha256'])) for b in pool['plans']]
    assert len(plans)==2 and all(t['split']=='train' for p in plans for t in p['tasks'])
    def run(label,args):
        with (root/(label+'.log')).open('w',encoding='utf-8') as log:
            r=subprocess.run(list(map(str,args)),cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert r.returncode==0,str(root/(label+'.log'))
    try:
        save(root/'progress.json',dict(stage='train-native-metrics'))
        cases={};totals={label:dict(battles=0,kills=0,dealt=0,received=0,death=0,goal=0,frames=0) for label in ('source','marginal')}
        for j in pool['jobs']:
            assert not j['error'] and j['mode']=='learned';label=spec['variants'][j['plan_index']]['label']
            folder=Path(j['root']);capture=read(pin(folder/'report.json'));manifest=read(pin(folder/'manifest.json'))
            assert capture['capture_complete'] and capture['provenance_valid'] and len(capture['results'])==1
            r=capture['results'][0];fixture=r['generated_fixture']
            assert r['capture_valid'] and r['dispatch_valid'] and r['seed_confirmed'] and fixture['split']=='train'
            assert fixture['engine_seed']==r['seed']==j['seed']
            plan=plans[j['plan_index']];assert plan['model_sha256']==manifest['model_weights_sha256'].lower()==sha(plan['model_path'])
            task=plan['tasks'][j['task_index']];assert task['split']=='train' and j['seed'] in task['seeds']
            damage=r['first_life']['damage'];metrics=dict(kills=sum(m['monster_kills'] for m in damage['by_mod']),
                dealt=sum(m['monster_health_damage'] for m in damage['by_mod']),received=damage['received_health_damage'],
                death=bool(r['death_stop'] or r['first_life']['minimum_health']<=0),goal=bool(r['goal_stop']),frames=r['actual_game_frames'])
            environment={k:manifest[k] for k in ('timescale','skill','release_game_frame','fixed_world_hold','monster_no_infighting',
                'standard_monster_spawn_height','post_frame_rng_reset','synchronous','stop_on_goal')}
            environment.update(client_sha256=manifest['client_sha256'],native_source_fingerprint=manifest['native_source_fingerprint'],
                exporter_sha256=manifest['exporter_sha256'],reward_sha256=manifest['reward_config_sha256'],runtime_files=runtime_bindings(r))
            key=(fixture['episode_id'],j['seed']);cases.setdefault(key,{})[label]=dict(fixture=fixture,environment=environment,metrics=metrics)
            totals[label]['battles']+=1
            for k,v in metrics.items():totals[label][k]+=v
        assert set(cases)=={(c['episode'],c['seed']) for c in spec['selected']};paired=[]
        for (episode,seed),pair in sorted(cases.items()):
            assert set(pair)=={'source','marginal'}
            assert pair['source']['fixture']==pair['marginal']['fixture'] and pair['source']['environment']==pair['marginal']['environment']
            paired.append(dict(episode=episode,seed=seed,metrics={k:v['metrics'] for k,v in pair.items()}))
        save(root/'native-metrics.json',dict(state='complete',version='combat_train_probe_metrics_v1',split='train',
            totals=totals,pairs=paired,inputs=pins,training_performed=False,quality_confirmation=False,promotion=False,
            scope='Selected known TRAIN replays; native outcomes only, not population quality. Causal prefix acceptance remains separate.'))
        processed=[]
        for index,v in enumerate(spec['variants']):
            label=v['label'];save(root/'progress.json',dict(stage='cuda-replay',label=label))
            execution=root/('execution-'+label+'.json')
            if execution.exists():
                e=read(execution);assert e['state']=='complete' and e['pool_sha256']==sha(root/'pool/report.json') and e['selected_plan_index']==index
            else:run('execution-'+label,[sys.executable,repo/'scripts/audit_registered_execution_actions.py','--pool',root/'pool','--plan-index',index,'--out',execution])
            directory=root/('processing-'+label)
            if directory.exists():
                assert (directory/'report.json').exists(),'Partial processing requires inspection; no hidden restart'
            else:run('processing-'+label,[sys.executable,repo/'scripts/process_registered_combat_pool_cuda.py','--pool',root/'pool','--plan-index',index,'--out',directory])
            meta=read(pin(directory/'report.json'))
            assert meta['state']=='complete' and meta['device']=='cuda' and not meta['training_performed'] and len(meta['cases'])==selected_count
            assert meta['pool_sha256']==sha(root/'pool/report.json') and meta['selected_plan_index']==index
            processed.append(dict(label=label,cases=selected_count,report_sha256=sha(directory/'report.json')))
        for p,h in pins.items():assert sha(Path(p))==h,'Input changed: '+p
        save(root/'complete.json',dict(state='complete',version='combat_joint_aim_context_probe_v1',native_cases=case_count,cuda_cases=case_count,
            spec_sha256=sha(root/'spec.json'),pool_sha256=sha(root/'pool/report.json'),processed=processed,
            native_metrics_sha256=sha(root/'native-metrics.json'),training_performed=False,promotion=False,quality_confirmation=False,
            scope='Known TRAIN diagnostic replays, no fitting. Causal prefix acceptance is separate.'))
        save(root/'progress.json',dict(stage='complete',native_cases=case_count,cuda_cases=case_count));print(totals)
    except BaseException as e:
        save(root/'progress.json',dict(stage='failed',error=str(e),exception=type(e).__name__));raise


if __name__=='__main__':main()
