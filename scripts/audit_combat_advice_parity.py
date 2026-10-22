"""Check instrumentation on/off on matched native learner trajectories, no NN execution."""
import argparse
import collections
import json
import pathlib

from evaluate_registered_combat_models import read, save, sha


def identity(value):
    return tuple(value[k] for k in ('map', 'connection', 'spawncount', 'actor', 'life', 'frame'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evaluation', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists()
    root = args.evaluation.resolve()
    complete = read(root/'complete.json'); assert complete['state'] == 'complete'
    pool_path = root/'pool/report.json'; pool = read(pool_path)
    assert pool['state'] == 'complete' and pool['source_unchanged']
    plans = [read(p['path']) for p in pool['plans']]
    assert len(plans) == 2 and not plans[0].get('record_combat_advice') and plans[1]['record_combat_advice']
    assert plans[0]['model_sha256'] == plans[1]['model_sha256']
    grouped = collections.defaultdict(dict)
    pins = {str(pool_path): sha(pool_path)}
    for job in pool['jobs']:
        assert not job['error']
        report_path = pathlib.Path(job['root'])/'report.json'; report = read(report_path)
        pins[str(report_path)] = sha(report_path)
        assert report['capture_complete'] and report['provenance_valid']
        result = report['results'][0]; assert result['capture_valid']
        trace = pathlib.Path(result['root'])/'bot.jsonl'; pins[str(trace)] = sha(trace)
        with trace.open(encoding='utf-8-sig') as stream:
            rows = [json.loads(line) for line in stream]
        indexed = {identity(r['combat_policy']['observation']['identity']): r for r in rows}
        assert len(indexed) == len(rows)
        label = 'advice-off' if job['plan_index'] == 0 else 'advice-on'
        sequence = root/('processing-'+label)/f"case-{job['job']:04d}"/'native/sequence.jsonl'
        pins[str(sequence)] = sha(sequence)
        with sequence.open(encoding='utf-8') as stream: exported = [json.loads(line) for line in stream]
        grouped[(job['task_index'], job['seed'])][job['plan_index']] = (indexed, result, exported)
    cases = []; advice_counts = collections.Counter()
    for (task, seed), pair in sorted(grouped.items()):
        assert set(pair) == {0, 1}
        off, roff, foff = pair[0]; on, ron, fon = pair[1]
        assert foff == fon and foff, ('canonical model inputs or causal memory', seed)
        assert not any('combat_advice' in r for r in off.values())
        # The wrapper may emit extra rules/death frames after the native
        # terminal event. They are not model inputs or supervised learner rows.
        all_advice = {k:r for k,r in on.items() if 'combat_advice' in r}
        advised = {k: r for k, r in all_advice.items() if r['combat_policy']['selection']['owner'] == 'provider'}
        assert advised and set(advised) <= set(off)
        # Native entity slots can differ during setup. Build a consistent
        # bijection only from same observed track/class/pose at matched frames.
        remap = {}
        for k, row in advised.items():
            left = off[k]['combat_policy']['observation']['enemies']
            right = row['combat_policy']['observation']['enemies']
            assert len(left) == len(right)
            for le, re in zip(left, right):
                for field in ('observed_track', 'class', 'relative'):
                    assert le.get(field) == re.get(field), ('enemy identity binding', seed, k, field)
                assert re['id'] not in remap or remap[re['id']] == le['id']
                remap[re['id']] = le['id']
        assert len(set(remap.values())) == len(remap)
        def normalize(value):
            if isinstance(value, list): return [normalize(v) for v in value]
            if not isinstance(value, dict): return value
            return {k: (remap.get(v,v) if k in ('id','entity','target_entity','aim_entity') and isinstance(v,int)
                        else 0 if k == 'observation_age_ms' and 0 <= v <= 300 else normalize(v))
                    for k,v in value.items()}
        def normalize_off(value):
            if isinstance(value, list): return [normalize_off(v) for v in value]
            if not isinstance(value, dict): return value
            return {k: 0 if k == 'observation_age_ms' and 0 <= v <= 300 else normalize_off(v) for k,v in value.items()}
        for k, row in advised.items():
            left, right = off[k], row
            lc, rc = left['combat_policy'], right['combat_policy']
            # Compare every model input and actual/sampled action exactly.
            assert normalize_off(lc['observation']) == normalize(rc['observation']), ('observation', seed, k)
            for field in ('proposed', 'applied', 'proposed_command', 'applied_command', 'command_changed'):
                assert lc[field] == normalize(rc[field]), (field, seed, k)
            for field in ('candidate', 'candidate_command', 'guarded_command', 'sample', 'owner', 'mode'):
                assert lc['selection'].get(field) == normalize(rc['selection'].get(field)), (field, seed, k)
            assert left['sent_command'] == right['sent_command'], ('sent_command', seed, k)
            a = row['combat_advice']; assert identity(a['identity']) == k
            assert a['version'] == 'reactive_combat_advice_v2'
            advice_counts['available' if a['available'] else a['reason']] += 1
            if a['available']:
                assert not a['recommended_action']['weapon']
                assert identity(a['recommended_action']['identity']) == k
        for field in ('goal_stop', 'death_stop'):
            # observed_frame is asynchronous polling of an already completed
            # native event, not the kill/death frame. Keep the latter exact.
            normalized = lambda r: {k:v for k,v in r[field].items() if k != 'observed_frame'} if r[field] else None
            assert normalized(roff) == normalized(ron), ('native outcome', seed, field)
        for field in ('minimum_health', 'damage', 'incoming_by_mod', 'incoming_by_attacker_class', 'outgoing_by_target_class'):
            assert roff['first_life'][field] == ron['first_life'][field], ('native damage', seed, field)
        # Every released learned command must be included, not a cherry-picked subset.
        owned = {k for k, r in on.items() if r['combat_policy']['selection']['owner'] == 'provider'}
        assert owned <= set(advised), ('unadvised learned frame', seed)
        off_owned = {k for k, r in off.items() if r['combat_policy']['selection']['owner'] == 'provider'}
        assert owned == off_owned, ('learned frame set differs', seed)
        cases.append(dict(task_index=task, seed=seed, advice_frames=len(advised), learned_frames=len(owned),
                          goal_stop=ron['goal_stop'], death_stop=ron['death_stop'],
                          canonical_input_rows=len(foff), entity_slot_mapping=remap,
                          wrapper_game_frames_off=roff['actual_game_frames'], wrapper_game_frames_on=ron['actual_game_frames']))
    assert advice_counts['available'] > 0
    for path, digest in pins.items(): assert sha(path) == digest
    save(args.out, dict(state='complete', version='combat_advice_parity_v1', cases=cases,
        counts=dict(advice_counts), native_cases=len(pool['jobs']), matched_observations_and_commands=True,
        inputs=[dict(path=p, sha256=h) for p,h in pins.items()], script_sha256=sha(__file__),
        neural_execution=False, training_performed=False, promotion=False,
        scope='Canonical Go model features and causal memory exact; sampled/applied/sent commands exact. Raw observations/actions allow a verified observed-track/class/pose entity-slot bijection and fresh wall-age differences only. Native terminal event and damage exact; setup/teardown and asynchronous observed_frame polling excluded. Instrumentation audit only; hypothetical expert quality is not established.'))
    print(args.out)


if __name__ == '__main__': main()
