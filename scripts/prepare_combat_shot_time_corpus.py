"""Pack verified current inputs and offline SG shot-time labels separately."""
import argparse
from collections import Counter
import gzip
import json
import shutil
from pathlib import Path
import numpy as np
from evaluate_registered_combat_models import read, save, sha
from audit_combat_contact_loss_probe import load_steps
from read_combat_shotgun_telemetry import parse


def compressed_copy(source, destination):
    with Path(source).open('rb') as src, destination.open('wb') as dst:
        with gzip.GzipFile(filename='', mode='wb', fileobj=dst, compresslevel=6, mtime=0) as encoded:
            shutil.copyfileobj(src, encoded)


def diagnostic_worlds(variants):
    # Outcome-independent: first site's offset-one world on each map; paired
    # controllers always belong to the same split. These are TRAIN-domain
    # diagnostic worlds, not globally unseen maps or a policy-quality test.
    result = set()
    maps = sorted({v['context']['map'] for v in variants})
    for name in maps:
        candidates = [v['context']['seed'] for v in variants
                      if v['context']['map'] == name and v['context']['fresh_offset'] == 1]
        assert candidates
        result.add((name, min(candidates)))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collection', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--split-reference', type=Path, help='Inherit pinned diagnostic world IDs; new worlds remain TRAIN')
    a = ap.parse_args(); assert not a.out.exists()
    root = a.collection.resolve()
    spec, pool, admission = (read(root/p) for p in ('spec.json', 'pool/report.json', 'acceptance.json'))
    assert pool['state'] == 'complete' and pool['source_unchanged']
    assert admission['state'] == 'passed' and admission['numerical_verification_complete']
    assert len(spec['variants']) == pool['usable_captures'] == spec['native_budget'] == spec['cuda_budget']
    assert all(v['mode'] == 'learned' and v['context']['loadout'] == 'shotgun' for v in spec['variants'])
    engine = spec['diagnostic_engine']; assert engine['telemetry'] == 'sg_trigger_pellet_pose_v1'
    if a.split_reference:
        reference = read(a.split_reference)
        assert reference['state'] == 'passed' and reference['version'] == 'combat_sg_shot_time_corpus_v1'
        diagnostic = {(e['map'], e['seed']) for e in reference['entries'] if e['split'] == 'diagnostic'}
        assert diagnostic
    else:
        diagnostic = diagnostic_worlds(spec['variants'])
    jobs = {j['plan_index']: j for j in pool['jobs']}
    a.out.mkdir(parents=True)
    entries, totals, coverage = [], Counter(), Counter()
    pins = {str(root/p): sha(root/p) for p in ('spec.json', 'pool/report.json', 'acceptance.json')}
    if a.split_reference:
        pins[str(a.split_reference.resolve())] = sha(a.split_reference)
    vertical = {'release': 0, 'jump': 1, 'crouch': 2}
    for v in spec['variants']:
        i = v['index']; case = root/f'case-{i:02d}'
        meta, proof = read(case/'cuda/report.json'), read(case/'cuda/cuda-verification.json')
        assert proof['state'] == 'passed' and proof['device'] == 'cuda'
        assert meta['model_sha256'] == v['model_sha256'] == sha(v['model'])
        assert sha(case/'cuda/cuda-verification.json') == meta['cuda_verification_sha256']
        native = next((case/'native').glob('replay-*/steps.jsonl'))
        assert sha(native) == meta['source_sha256'][str(native.resolve())]
        steps, rollout = load_steps(native), load_steps(case/'cuda/rollout.jsonl')
        assert len(rollout) == meta['rows'] and sha(case/'cuda/rollout.jsonl') == meta['rollout_sha256']
        by_frame = {s['observation']['identity']['frame']: (j, s) for j, s in enumerate(steps)}
        assert len(by_frame) == len(steps)
        report = read(Path(jobs[i]['root'])/'report.json')['results'][0]
        runtime = Path(report['root'])/'runtime'
        for name in ('q2ded.exe', 'baseq2/game.dll'):
            assert sha(runtime/name) == engine['binaries'][name]
        log = Path(report['root'])/'server.log'
        with log.open(encoding='utf-8-sig') as stream:
            shots = parse(stream)
        assert shots, 'Shotgun-start collection has no trigger'
        d = a.out/f'case-{i:02d}'; d.mkdir()
        frames = [r['frame'] for r in rollout]
        frame_row = {f:j for j, f in enumerate(frames)}
        assert len(frame_row) == len(rollout)
        actions, applied, next_delta = [], [], []
        for r in rollout:
            _, s = by_frame[r['frame']]
            assert s['owner'] == 'provider' and s['sample'] == r['sample']
            for store, key in ((actions, 'action'), (applied, 'applied_action')):
                act = s[key]
                store.append([act['forward'], act['side'], act['yaw_delta_degrees'], act['pitch_delta_degrees'],
                              int(act['attack']), vertical[act['vertical']]])
            o, n = s['observation'], s['next_observation']
            next_delta.append([n['position'][k]-o['position'][k] for k in range(3)] if n is not None else [np.nan]*3)
        counts = np.zeros(len(rollout), dtype=np.int16)
        for shot in shots:
            frame = shot['window']['begin_frame']
            assert frame in by_frame
            j, s = by_frame[frame]
            shot['native_trace_row'] = j
            shot['input_row'] = frame_row.get(frame)
            shot['selected_target'] = s['action'].get('target_entity')
            shot['selected_vertical'] = s['action']['vertical']
            shot['policy_label_eligible'] = frame in frame_row
            if frame in frame_row:
                counts[frame_row[frame]] += 1
                totals['eligible_triggers'] += 1
            else:
                totals['nonprovider_or_terminal_triggers'] += 1
            coverage['native_phase_'+str(shot['native_phase_offset'])] += 1
            coverage['requested_'+s['action']['vertical']] += 1
            coverage['shot_ducked_'+shot['begin']['ducked']] += 1
            coverage['shot_grounded_'+shot['begin']['grounded']] += 1
        features = np.asarray([r['features'] for r in rollout], dtype=np.float32)
        assert features.shape == (len(rollout), 1554) and np.isfinite(features).all()
        pack = d/'frames.npz'
        np.savez_compressed(pack, inputs_features=features, inputs_requested_action=np.asarray(actions, dtype=np.float32),
             inputs_sampled_weapon=np.asarray([r['sample'].get('weapon', 0) for r in rollout], dtype=np.int16),
             inputs_sampled_target=np.asarray([r['sample'].get('target', 0) for r in rollout], dtype=np.int16),
             inputs_sampled_aim_mode=np.asarray([r['sample'].get('aim_mode', 0) for r in rollout], dtype=np.int16),
             frames=np.asarray(frames, dtype=np.int32),
             labels_applied_action=np.asarray(applied, dtype=np.float32),
             labels_next_position_delta=np.asarray(next_delta, dtype=np.float32),
             labels_next_valid=np.asarray([by_frame[f][1]['next_observation'] is not None for f in frames], dtype=np.bool_),
             labels_shot_count=counts)
        labels = d/'shots.jsonl.gz'
        with gzip.open(labels, 'wt', encoding='utf-8', compresslevel=6) as stream:
            for shot in shots:
                stream.write(json.dumps(shot, allow_nan=False)+'\n')
        compressed_native, compressed_sequence = d/'native.jsonl.gz', d/'sequence.jsonl.gz'
        compressed_copy(native, compressed_native)
        compressed_copy(case/'cuda/sequence.jsonl', compressed_sequence)
        save(d/'verification.json', dict(model_sha256=v['model_sha256'], cuda_verification=proof,
             cuda_verification_sha256=sha(case/'cuda/cuda-verification.json'), source_meta=meta))
        # Serialized arrays are caches. Original observations/context and their
        # immutable native trace remain available for other feature encoders.
        world = (v['context']['map'], v['context']['seed'])
        entry = dict(index=i, map=world[0], seed=world[1], episode=v['context']['episode'], controller=v['label'],
             split='diagnostic' if world in diagnostic else 'train', source_model_sha256=v['model_sha256'],
             rows=len(rollout), triggers=len(shots), pellets=sum(len(s['pellets']) for s in shots),
             pack=str(pack.resolve()), pack_sha256=sha(pack), labels=str(labels.resolve()), labels_sha256=sha(labels),
             native_trace=str(native.resolve()), native_trace_sha256=sha(native),
             compressed_native=str(compressed_native.resolve()), compressed_native_sha256=sha(compressed_native),
             sequence=str((case/'cuda/sequence.jsonl').resolve()), sequence_sha256=sha(case/'cuda/sequence.jsonl'),
             compressed_sequence=str(compressed_sequence.resolve()), compressed_sequence_sha256=sha(compressed_sequence),
             verification=str((d/'verification.json').resolve()), verification_sha256=sha(d/'verification.json'),
             source_log=str(log), source_log_sha256=sha(log))
        entries.append(entry)
        for key in ('rows', 'triggers', 'pellets'):
            totals[key] += entry[key]
        pins[str(case/'cuda/cuda-verification.json')] = sha(case/'cuda/cuda-verification.json')
        if (i+1) % 8 == 0:
            print(dict(prepared=i+1, cases=len(spec['variants'])), flush=True)
    worlds = {s: {(e['map'], e['seed']) for e in entries if e['split'] == s} for s in ('train', 'diagnostic')}
    assert worlds['train'] and (a.split_reference or worlds['diagnostic']) and not worlds['train'] & worlds['diagnostic']
    save(a.out/'report.json', dict(version='combat_sg_shot_time_corpus_v1', state='passed', entries=entries,
         native_cases=len(entries), unique_worlds=len(worlds['train'] | worlds['diagnostic']),
         split_worlds={s:len(w) for s,w in worlds.items()}, totals=dict(totals), shot_coverage=dict(coverage),
         action_columns=['forward', 'side', 'yaw_delta_degrees', 'pitch_delta_degrees', 'attack', 'vertical_0release_1jump_2crouch'],
         input_arrays=['inputs_features', 'inputs_requested_action', 'inputs_sampled_weapon', 'inputs_sampled_target', 'inputs_sampled_aim_mode'],
         label_arrays=['labels_applied_action', 'labels_next_position_delta', 'labels_next_valid', 'labels_shot_count'],
         feature_version='combat_features_v13_memory_static_routes', engine=engine, inputs=pins,
         parser_sha256=sha(Path(__file__).with_name('read_combat_shotgun_telemetry.py')),
         preparer_sha256=sha(Path(__file__)), ppo_eligible=False, policy_runtime_inputs_include_labels=False,
         training_performed=False, promotion=False, split_reference=str(a.split_reference.resolve()) if a.split_reference else None,
         scope='Action-effect and shot-time auxiliary dataset. All outcomes retained. Shot groundtruth includes hidden poses; labels NEVER runtime inputs. Controller lineages differ beyond aim. Diagnostic worlds are TRAIN-domain seed holdouts, not globally held-out maps or a policy-quality test. With split-reference, diagnostic IDs are inherited and new worlds are TRAIN.'))
    print(dict(totals=dict(totals), split_worlds={s:len(w) for s,w in worlds.items()}, coverage=dict(coverage)), flush=True)


if __name__ == '__main__':
    main()
