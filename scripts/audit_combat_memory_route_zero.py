"""Scalar native zero-identity audit; no neural evaluation."""
import argparse
import copy
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha
from audit_combat_contact_loss_probe import equivalent, load_steps


def canonical_metrics(result):
    data = copy.deepcopy({k: result[k] for k in ('first_life','goal_stop','death_stop')})
    # Summary telemetry enumerates a Go map without order. This is NOT policy
    # input ordering: replay observations keep their exact slot order below.
    for frame in data['first_life'].get('last_frames', []):
        frame['visible_enemies'] = sorted(frame.get('visible_enemies') or [], key=lambda e:e['id'])
    return data


def audit(root, output):
    assert not output.exists()
    spec, pool = read(root/'spec.json'), read(root/'pool/report.json')
    assert spec['version']=='combat_memory_route_zero_native_v1' and len(spec['variants'])==6
    assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==6
    batch = read(root/'cuda-batch/report.json')
    assert batch['state']=='complete' and len(batch['tasks'])==6
    jobs={j['plan_index']:j for j in pool['jobs']}
    pairs=[]
    for ci in range(3):
        members=[v for v in spec['variants'] if v['context_index']==ci]
        assert [v['label'] for v in members]==['source','zero']
        traces, results, route_frames = [], [], []
        for v in members:
            d=root/f'case-{v["index"]:02d}'
            proof,meta=read(d/'cuda/cuda-verification.json'),read(d/'cuda/report.json')
            assert proof['state']=='passed' and proof['device']=='cuda'
            assert sha(v['model'])==v['model_sha256']==meta['model_sha256']
            path=next((d/'native').glob('replay-*/steps.jsonl'))
            assert sha(path)==meta['source_sha256'][str(path.resolve())]
            steps=load_steps(path)
            route_frames.append(sum(s['observation'].get('remembered_static_routes') is not None for s in steps))
            for s in steps:
                s['observation'].pop('remembered_static_routes',None)
            traces.append(steps)
            capture=read(Path(jobs[v['index']]['root'])/'report.json')
            r=capture['results'][0]
            assert capture['capture_complete'] and capture['provenance_valid'] and r['capture_valid'] and r['dispatch_valid']
            assert r['generated_fixture']==v['context']['instance'] and r['seed_confirmed']
            results.append(r)
        ends=[r['first_life']['end_frame'] for r in results]
        exact=ends[0]==ends[1] and canonical_metrics(results[0])==canonical_metrics(results[1]) and equivalent(*traces,before=ends[0]+1)
        assert route_frames[0]==0 and route_frames[1]>0
        # A real native command difference must still fail admission.
        mutation=copy.deepcopy(traces[1])
        active=next(s for s in mutation if s['owner']=='provider' and s['observation']['identity']['frame']<ends[0])
        active['sent_command']['Side']+=1
        assert not equivalent(traces[0],mutation,before=ends[0]+1)
        pairs.append(dict(map=members[0]['context']['map'],seed=members[0]['context']['seed'],
            end_frames=ends,identity_exact=exact,route_observation_frames=route_frames,
            command_mutation_rejected=True,goal=bool(results[0]['goal_stop'])))
    report=dict(version='combat_memory_route_zero_native_acceptance_v2',state='passed' if all(p['identity_exact'] for p in pairs) else 'failed',
        pairs=pairs,native_cases=6,cuda_cases=6,training_performed=False,promotion=False,
        inputs={str((root/p).resolve()):sha(root/p) for p in ('spec.json','pool/report.json','cuda-batch/report.json')},
        scope='All commands, sampled outcomes and physical observations equal through task boundary. Only new route input excluded; only unordered summary enemy lists sorted. No quality improvement established.')
    save(output,report)
    assert report['state']=='passed'
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--probe',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args()
    print(audit(a.probe.resolve(),a.out.resolve())['state'])
