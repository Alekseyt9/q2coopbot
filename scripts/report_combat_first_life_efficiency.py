"""Measure a sealed training pool; distinguish skipped budget from wall speedup."""
import argparse
import pathlib
from process_combat_architecture_pool import read, save, sha


def report(capture, processing, output):
    protocol = read(processing/'protocol.json')
    complete = read(processing/'report.json')
    pool = read(capture/'pool/report.json')
    selected = read(processing/'selected-captures.json')
    assert complete['state'] == 'complete' and complete['protocol'] == protocol
    assert pathlib.Path(protocol['capture_root']).resolve() == capture
    assert protocol['pool_sha256'] == sha(capture/'pool/report.json')
    assert pool['state'] == 'complete' and pool['source_unchanged']
    assert not any(j['error'] for j in pool['jobs']), 'Report requires a complete unretried pool'
    assert len(selected) == len(pool['jobs']) == protocol['allocated_episodes']
    expected = {(j['plan_index'],j['task_index'],j['seed'],j['mode']):j for j in pool['jobs']}
    assert len(expected) == len(selected)
    episodes, sources = [], {}
    for entry in selected:
        key = (entry['plan'],entry['task'],entry['seed'],entry['mode'])
        assert key in expected
        job = expected.pop(key)
        root = pathlib.Path(entry['root']).resolve()
        assert root == pathlib.Path(job['root']).resolve()
        for filename,field in [('report.json','report_sha256'),('manifest.json','manifest_sha256')]:
            assert sha(root/filename) == entry[field]
            sources[str(root/filename)] = entry[field]
        native, manifest = read(root/'report.json'), read(root/'manifest.json')
        assert native['capture_complete'] and native['provenance_valid'] and len(native['results']) == 1
        result = native['results'][0]
        assert result['seed'] == entry['seed'] and all(result[k] for k in ('capture_valid','seed_confirmed','dispatch_valid','frame_budget_valid'))
        worker = pathlib.Path(result['root'])
        data = result['dataset']
        masks = data.get('reward_masks',{})
        assert data['reward_steps']+sum(masks.values()) == data['steps']
        frames, limit = result['actual_game_frames'], manifest['game_frames']
        assert 0 < frames <= limit and manifest['timescale'] == 2
        death = result.get('death_stop')
        skipped = 0
        if death:
            proof_path = worker/'dataset/death-stop-verification.json'
            proof = read(proof_path)
            assert proof['state'] == 'verified' and proof['receipt'] == death
            assert result['first_life']['end_reason'] == 'first_observed_death' and not result.get('goal_stop')
            assert frames < limit
            sources[str(proof_path)] = sha(proof_path)
            skipped = limit-frames
        episodes.append(dict(seed=entry['seed'],task=entry['task'],game_frames=frames,
            configured_frame_budget=limit,death_stop=bool(death),
            budget_frames_not_run_after_death=skipped,job_wall_seconds=job['wall_seconds'],
            exported_steps=data['steps'],rewarded_steps=data['reward_steps'],
            masked_after_first_life_steps=masks.get('after_first_life',0),
            trace_bytes=(worker/'bot.jsonl').stat().st_size,
            exported_steps_bytes=(worker/'dataset/steps.jsonl').stat().st_size))
    assert not expected
    fields = ('game_frames','budget_frames_not_run_after_death','job_wall_seconds','exported_steps',
              'rewarded_steps','masked_after_first_life_steps','trace_bytes','exported_steps_bytes')
    totals = {field:sum(e[field] for e in episodes) for field in fields}
    assert totals['game_frames'] == pool['actual_game_frames']
    result = dict(state='complete',episodes=episodes,totals=totals,
        death_stopped_episodes=sum(e['death_stop'] for e in episodes),
        slots=pool['slots'],timescale=2,pool_wall_seconds=pool['wall_seconds'],
        episodes_per_wall_second=len(episodes)/pool['wall_seconds'],
        inputs=dict(pool_sha256=sha(capture/'pool/report.json'),processing_report_sha256=sha(processing/'report.json'),
                    selected_captures_sha256=sha(processing/'selected-captures.json')),
        source_sha256=sources,
        scope='Observed completed native training pool. Unspent frame budget after verified death is a budget counterfactual, not measured wall-time savings. Goal stopping already existed. Different seeds prevent a causal speed/storage comparison with old collections; startup, export and source verification remain included in pool wall time.')
    for path,digest in sources.items():
        assert sha(path) == digest
    save(output,result)
    return {k:result[k] for k in ('state','totals','death_stopped_episodes','pool_wall_seconds')}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for name in ('capture','processing','out'):
        ap.add_argument('--'+name,type=pathlib.Path,required=True)
    a = ap.parse_args()
    print(report(a.capture.resolve(),a.processing.resolve(),a.out.resolve()))
