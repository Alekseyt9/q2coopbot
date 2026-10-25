"""Native zero-yaw migration parity; physical/action audit, no NN on CPU."""
import argparse
import copy
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from audit_combat_contact_loss_probe import equivalent,load_steps
from audit_combat_memory_route_zero import canonical_metrics


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--screen',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args();assert not a.out.exists();root=a.screen.resolve()
    spec=read(root/'spec.json');pool=read(root/'pool/report.json');batch=read(root/'cuda-batch/report.json')
    assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==spec['native_budget']==6
    assert batch['state']=='complete' and len(batch['tasks'])==spec['cuda_budget']==6
    jobs={j['plan_index']:j for j in pool['jobs']};pairs=[];total_active=0;pins={}
    for ci in range(spec['independent_worlds']):
        variants=[v for v in spec['variants'] if v['context_index']==ci]
        assert [v['label'] for v in variants]==['source','zero']
        traces=[];results=[];active=0
        for v in variants:
            model=read(v['model']);version=model['memory_route_movement']['version']
            assert version==('combat_memory_route_movement_v1' if v['label']=='source' else 'combat_memory_route_movement_yaw_v2')
            assert sha(v['model'])==v['model_sha256']
            d=root/f'case-{v["index"]:02d}';meta=read(d/'cuda/report.json');proof=read(d/'cuda/cuda-verification.json')
            assert proof['state']=='passed' and proof['device']=='cuda' and meta['memory_route_movement_version']==version
            assert meta['model_sha256']==v['model_sha256'] and meta['ppo_training_ready'] is False
            for name in ('cuda/report.json','cuda/cuda-verification.json'):pins[str((d/name).resolve())]=sha(d/name)
            steps_path=next((d/'native').glob('replay-*/steps.jsonl'));steps=load_steps(steps_path)
            pins[str(steps_path.resolve())]=sha(steps_path);traces.append(steps)
            if v['label']=='zero':
                for r in load_steps(d/'cuda/rollout.jsonl'):
                    x=r['features']
                    active+=bool(x[0]>0 and x[1489]==1 and not any(x[73+12*j] for j in range(8))
                        and any(x[881+30*j]==1 and x[1490+8*j]==1 and x[1491+8*j]==1 for j in range(8)))
            capture=Path(jobs[v['index']]['root']);report=read(capture/'report.json');r=report['results'][0]
            assert report['capture_complete'] and report['provenance_valid'] and r['capture_valid'] and r['dispatch_valid'] and r['seed_confirmed']
            assert r['generated_fixture']==v['context']['instance'];results.append(r)
        end=results[0]['first_life']['end_frame']
        assert end==results[1]['first_life']['end_frame']
        assert canonical_metrics(results[0])==canonical_metrics(results[1])
        assert equivalent(*traces,before=end+1)
        mutated=copy.deepcopy(traces[1]);step=next(s for s in mutated if s['owner']=='provider' and s['observation']['identity']['frame']<end)
        step['sent_command']['Side']+=1;assert not equivalent(traces[0],mutated,before=end+1)
        total_active+=active
        pairs.append(dict(map=variants[0]['context']['map'],seed=variants[0]['context']['seed'],
                          native_commands_samples_physical_outcomes_equivalent=True,active_rows=active,command_mutation_rejected=True))
    assert total_active>0,'No native active route evidence'
    save(a.out,dict(state='passed',version='combat_route_yaw_zero_native_parity_v1',pairs=pairs,route_active_rows=total_active,
        native_cases=6,cuda_cases=6,training_performed=False,promotion=False,inputs=pins,
        scope='Zero migration native physical/actions/sample parity with existing identity normalization; no trained yaw quality claim.'))
    print(dict(pairs=len(pairs),active_rows=total_active,state='passed'),flush=True)


if __name__=='__main__':main()
