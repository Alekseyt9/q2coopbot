"""Observed one-step motion labels, canonical current features, real current action.

No teacher advice, future action, reward or native hidden pose enters an input.
NumPy is used for compact storage only; neural computation belongs to CUDA.
"""
import argparse
import collections
import json
import math
import pathlib

from evaluate_registered_combat_models import read, save, sha

FIELDS = ('map','connection','spawncount','actor','life','frame')


def identity(o):
    return tuple(o['identity'][k] for k in FIELDS)


def local(v, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return [v[0]*c+v[1]*s, -v[0]*s+v[1]*c, v[2]]


def subtract(a,b): return [x-y for x,y in zip(a,b)]


def example(step, trace, features):
    o, n = step['observation'], step['next_observation']
    if step['owner'] != 'provider': return None, 'not_provider'
    if n is None: return None, 'missing_successor'
    if o['health'] <= 0 or n['health'] <= 0: return None, 'dead_endpoint'
    if identity(o)[:-1] != identity(n)[:-1] or n['identity']['frame'] != o['identity']['frame']+1:
        return None, 'reset_or_gap'
    native, dispatch = step.get('native_step'), step.get('server_execution') or {}
    if not native or not dispatch.get('matched') or not dispatch.get('window_exclusive') or dispatch.get('recovery_commands') != 0:
        return None, 'unverified_dispatch'
    if (native['begin_frame'] != o['identity']['frame'] or native['end_frame'] != n['identity']['frame']
            or native['spawncount'] != o['identity']['spawncount'] or native['actor'] != o['identity']['actor']
            or native['sequence'] != step['client_sequence']):
        return None, 'native_identity_or_duration'
    cmd = step['sent_command']
    if cmd['Msec'] != 100: return None, 'unsupported_msec'
    assert trace['combat_policy']['observation'] == o and trace['sent_command'] == cmd
    a = step['applied_action']; assert a['identity'] == o['identity']
    assert a['forward'] == cmd['Forward']/400 and a['side'] == cmd['Side']/400
    assert a['attack'] == bool(cmd['Buttons'] & 1)
    yaw = o['view_angles'][1]*2*math.pi/65536
    target = [local(subtract(n['position'],o['position']),yaw)] + [[0.]*3 for _ in range(8)]
    cv = [local([v*.1 for v in o['velocity']],yaw)] + [[0.]*3 for _ in range(8)]
    motion_mask = [True]+[False]*8; velocity_known = [True]+[False]*8
    present = [0.]*8; present_mask = [False]*8
    clear = [0.]*8; clear_mask = [False]*8; clear_cv = [0.]*8; clear_cv_mask = [False]*8
    classes = ['']*8
    future = {(e['id'],e.get('observed_track')):e for e in n['enemies']}
    assert len(future) == len(n['enemies'])
    enemies = sorted(o['enemies'], key=lambda e:e['distance'])[:8]
    for i,e in enumerate(enemies):
        if e['id'] <= 0 or not e.get('observed_track'): continue
        classes[i] = e['class']; present_mask[i] = True
        velocity = e.get('observed_velocity')
        if velocity is not None:
            velocity_known[i+1] = True; cv[i+1] = local([v*.1 for v in velocity],yaw)
        if e.get('clear_shot') is not None:
            clear_cv[i] = float(e['clear_shot']); clear_cv_mask[i] = True
        nxt = future.get((e['id'],e['observed_track']))
        if nxt is None: continue
        assert nxt['class'] == e['class']
        present[i] = 1.; motion_mask[i+1] = True
        # World displacement, not apparent relative displacement from ego motion.
        delta = [n['position'][j]+nxt['relative'][j]-o['position'][j]-e['relative'][j] for j in range(3)]
        target[i+1] = local(delta,yaw)
        if nxt.get('clear_shot') is not None:
            clear[i] = float(nxt['clear_shot']); clear_mask[i] = True
    pmove = trace.get('pmove_ground_prediction')
    pmove_mask = bool(pmove and pmove['initial_grounded'] and not pmove['support_ambiguous'])
    pmove_delta = local(subtract(pmove['position'],o['position']),yaw) if pmove_mask else [0.]*3
    action = [a['forward'],a['side'],a['yaw_delta_degrees']/180,a['pitch_delta_degrees']/180,
              cmd['Up']/400,float(a['attack'])]
    assert len(features) == 1147 and all(math.isfinite(v) for v in features+action)
    return dict(x=features+action, target=[[v/64 for v in xyz] for xyz in target],
        cv=[[v/64 for v in xyz] for xyz in cv], motion_mask=motion_mask,velocity_known=velocity_known,
        present=present,present_mask=present_mask,clear=clear,clear_mask=clear_mask,
        clear_cv=clear_cv,clear_cv_mask=clear_cv_mask,pmove=[v/64 for v in pmove_delta],pmove_mask=pmove_mask,
        action_sensitive=bool(cmd['Forward'] or cmd['Side'] or cmd['Up']),classes=classes), None


def main():
    ap = argparse.ArgumentParser()
    for name in ('corpus','out'): ap.add_argument('--'+name,type=pathlib.Path,required=True)
    args = ap.parse_args(); assert not args.out.exists()
    import numpy as np
    root = args.corpus.resolve(); pins = {}
    def pin(path,expected=None):
        path = pathlib.Path(path).resolve(); digest = sha(path); assert expected is None or expected == digest,str(path)
        pins[str(path)] = digest; return path
    closed = read(pin(root/'complete.json')); assert closed['state'] == 'complete' and not closed['training_performed']
    protocol = read(pin(root/'protocol.json',closed['protocol_sha256']))
    pool = read(pin(root/'pool/report.json',closed['pool_sha256']))
    assert pool['state'] == 'complete' and pool['source_unchanged'] and pool['usable_captures'] == len(pool['jobs'])
    assert all(not j['error'] for j in pool['jobs'])
    args.out.mkdir(parents=True); splits = {}; seen = set(); classes = {}
    for plan_index, assignment in enumerate(protocol['assignments']):
        split = assignment['label']; assert split in ('train','development')
        plan = read(pin(assignment['plan'],assignment['plan_sha256']))
        assert all(t['split'] == ('train' if split == 'train' else 'validation') for t in plan['tasks'])
        receipt = next(p for p in closed['processed'] if p['label'] == split)
        processed = read(pin(root/('processing-'+split)/'report.json',receipt['sha256']))
        assert processed['state'] == 'complete' and processed['device'] == 'cuda' and not processed['training_performed']
        assert processed['pool_sha256'] == closed['pool_sha256'] and processed['selected_plan_index'] == plan_index
        rows = []; keys = []; cases = []; excluded = collections.Counter(); seeds = set()
        for job in sorted((j for j in pool['jobs'] if j['plan_index'] == plan_index),key=lambda j:j['job']):
            seeds.add(job['seed']); case_index = len(cases)
            capture = read(pin(pathlib.Path(job['root'])/'report.json'))
            assert capture['capture_complete'] and capture['provenance_valid'] and len(capture['results']) == 1
            result = capture['results'][0]; assert result['capture_valid']
            trace = pin(pathlib.Path(result['root'])/'bot.jsonl')
            with trace.open(encoding='utf-8-sig') as stream:
                raw_rows = [json.loads(l) for l in stream]
            raw = {identity(r['combat_policy']['observation']):r for r in raw_rows}; assert len(raw) == len(raw_rows)
            directory = root/('processing-'+split)/f"case-{job['job']:04d}"/'native'
            native_report = read(pin(directory/'report.json'))
            cuda_report = read(pin(directory.parent/'cuda/report.json'))
            cuda_proof = read(pin(directory.parent/'cuda/cuda-verification.json'))
            assert native_report['feature_version'] == cuda_report['feature_version'] == 'combat_features_v10'
            assert cuda_report['numerical_verification'] == 'cuda_verified_v1'
            assert cuda_proof['state'] == 'passed' and cuda_proof['device'] == 'cuda'
            assert native_report['native_rollout_sha256'] == cuda_proof['native_rollout_sha256'] == cuda_report['native_rollout_sha256']
            assert native_report['sequence_sha256'] == cuda_report['sequence_sha256']
            assert native_report['model_sha256'] == cuda_proof['model_sha256'] == processed['model_sha256']
            original_inputs = {str(pathlib.Path(p).resolve()):h for p,h in native_report['source_sha256'].items()}
            sequence = pin(directory/'sequence.jsonl',native_report['sequence_sha256'])
            for source_file in (trace,pathlib.Path(job['root'])/'report.json',directory/'replay-0/steps.jsonl',pathlib.Path(result['root'])/'server.log'):
                source_file = source_file.resolve();pin(source_file,original_inputs[str(source_file)])
            with sequence.open(encoding='utf-8') as stream:
                seq_rows = list(map(json.loads,stream))
            seq = {(r['index'],r['frame']):r for r in seq_rows}; assert len(seq) == len(seq_rows)
            steps = pin(directory/'replay-0/steps.jsonl'); count = 0; reasons = collections.Counter()
            with steps.open(encoding='utf-8-sig') as stream:
                for step in map(json.loads,stream):
                    o, n = step['observation'],step['next_observation']
                    if n is None:
                        reasons['missing_successor'] += 1; continue
                    current, successor = raw.get(identity(o)),raw.get(identity(n))
                    if current is None or successor is None:
                        reasons['missing_real_trace_endpoint'] += 1; continue
                    assert current['combat_policy']['observation'] == o and successor['combat_policy']['observation'] == n
                    f = seq.get((step['index'],o['identity']['frame']))
                    if f is None: reasons['not_canonical_provider_frame'] += 1; continue
                    assert f['seed'] == job['seed']
                    row, reason = example(step,current,f['features'])
                    if reason: reasons[reason] += 1; continue
                    for name in row['classes']:
                        if name and name not in classes: classes[name] = len(classes)+1
                    row['class_id'] = [classes.get(c,0) for c in row.pop('classes')]
                    row.update(case_index=case_index,seed=job['seed'],frame=o['identity']['frame'])
                    rows.append(row); keys.append((job['seed'],identity(o))); count += 1
            assert count > 0
            excluded.update(reasons)
            cases.append(dict(job=job['job'],seed=job['seed'],map=plan['tasks'][job['task_index']]['episode']['map'],
                              episode=plan['tasks'][job['task_index']]['episode']['id'],rows=count,excluded=dict(reasons)))
        assert len(keys) == len(set(keys)) and not seen.intersection(seeds); seen.update(seeds)
        bools = ('motion_mask','velocity_known','present_mask','clear_mask','clear_cv_mask','pmove_mask','action_sensitive')
        ints = ('case_index','seed','frame','class_id')
        arrays = {k:np.array([r[k] for r in rows],dtype=np.bool_ if k in bools else np.int64 if k in ints else np.float32) for k in rows[0]}
        for k,v in arrays.items(): assert np.isfinite(v).all(),k
        assert arrays['x'].shape == (len(rows),1153) and arrays['target'].shape == (len(rows),9,3)
        path = args.out/(split+'.npz'); np.savez_compressed(path,**arrays)
        with np.load(path,allow_pickle=False) as restored:
            assert set(restored.files) == set(arrays) and all(np.array_equal(restored[k],v) for k,v in arrays.items())
        splits[split] = dict(path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size,rows=len(rows),
            cases=cases,seeds=sorted(seeds),excluded=dict(excluded),motion_labels=int(arrays['motion_mask'].sum()),
            track_labels=int(arrays['present_mask'].sum()),track_disappearances=int(((arrays['present']==0)&arrays['present_mask']).sum()),
            pmove_rows=int(arrays['pmove_mask'].sum()),action_sensitive_rows=int(arrays['action_sensitive'].sum()))
        splits[split].update(clear_labels=int(arrays['clear_mask'].sum()),
            clear_positive=int((arrays['clear']*arrays['clear_mask']).sum()),
            conditional_clear_single_class=not bool(((arrays['clear']==0)&arrays['clear_mask']).any()))
    assert set(splits) == {'train','development'}
    for p,h in pins.items(): assert sha(p) == h
    save(args.out/'complete.json',dict(state='complete',version='combat_observed_future_dataset_v1',feature_version='combat_features_v10',
        inputs_width=1153,current_features_width=1147,dt_game_seconds=.1,displacement_scale=64,classes=classes,splits=splits,
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),training_performed=False,neural_execution=False,
        scope='Adjacent real alive observations, same identity, exclusive native100ms command. Current canonical features plus current applied action only; no future actions, advice, rewards or native hidden poses. Motion in initial yaw forward/left/up axes; entity+track joins. Float32 compact storage after canonical feature binding.'))
    print(args.out/'complete.json')


if __name__ == '__main__': main()
