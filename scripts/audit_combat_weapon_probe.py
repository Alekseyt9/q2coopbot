"""Scalar native prefix, cold-repeat and weapon acknowledgement audit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from evaluate_registered_combat_models import read, save, sha
from audit_combat_joint_aim_grid import physical_equal_with_entity_bijection


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--probe', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); assert not a.out.exists(); root = a.probe.resolve()
    seal = read(root/'complete.json'); spec = read(root/'spec.json'); pool = read(root/'pool/report.json')
    assert seal['native_cases'] == seal['cuda_cases'] == 24 and sha(root/'spec.json') == seal['spec_sha256']
    assert sha(root/'pool/report.json') == seal['pool_sha256'] and pool['source_unchanged']
    assert sha(root/'cuda-batch/report.json') == seal['cuda_receipt_sha256']
    jobs = {j['plan_index']: j for j in pool['jobs']}
    loaded = []
    for v in spec['variants']:
        i = v['plan_index']; directory = root/f'case-{i:02d}'; capture = Path(jobs[i]['root'])
        meta = read(directory/'cuda/report.json'); proof = read(directory/'cuda/cuda-verification.json')
        assert proof['state'] == 'passed' and proof['device'] == 'cuda'
        assert sha(directory/'cuda/cuda-verification.json') == meta['cuda_verification_sha256']
        assert meta['model_sha256'] == v['model_sha256'] == sha(v['model'])
        if v['label'] != 'source':
            assert meta['diagnostic_only'] and not meta['ppo_training_ready'] and not meta['ordinary_training_eligible']
            assert proof['diagnostic_weapon_probe']['root_rows'] == 1
        path = next((directory/'native').glob('replay-*/steps.jsonl'))
        assert sha(path) == meta['source_sha256'][str(path.resolve())]
        steps = [json.loads(line) for line in path.open(encoding='utf-8-sig')]
        controlled = [s for s in steps if s['owner'] == 'provider' and s.get('sample') and s.get('native_step')
            and s['observation']['identity']['life'] == 1 and s['observation']['health'] > 0
            and (s.get('server_execution') or {}).get('matched')
            and (s.get('server_execution') or {}).get('window_exclusive')
            and (s.get('server_execution') or {}).get('recovery_commands') == 0]
        rows_path = directory/'cuda/rollout.jsonl'
        assert sha(rows_path) == meta['rollout_sha256']
        rows = {r['frame']: r for r in map(json.loads, rows_path.open(encoding='utf-8-sig'))}
        report = read(capture/'report.json'); assert report['capture_complete'] and report['provenance_valid']
        result = report['results'][0]; life = result['first_life']; damage = life['damage']
        metrics = dict(goal=bool(result['goal_stop']), death=bool(result['death_stop']), end_frame=life['end_frame'],
            kills=sum(q['monster_kills'] for q in damage['by_mod']),
            dealt=sum(q['monster_health_damage'] for q in damage['by_mod']), received=damage['received_health_damage'])
        loaded.append(dict(variant=v, steps=controlled, all_steps=steps, rows=rows, result=result, metrics=metrics,
                           native_root=Path(result['root'])))

    fields = ('sampling_seed', 'latent', 'attack', 'vertical', 'weapon', 'target', 'aim_mode',
              'memory', 'aim_candidate', 'log_probability', 'value')
    def equivalent(left, right, before=None):
        aa = [s for s in left['steps'] if before is None or s['observation']['identity']['frame'] < before]
        bb = [s for s in right['steps'] if before is None or s['observation']['identity']['frame'] < before]
        forward, reverse = {}, {}
        if len(aa) != len(bb): return False
        for x, y in zip(aa, bb):
            f = x['observation']['identity']['frame']
            if f != y['observation']['identity']['frame']: return False
            if not physical_equal_with_entity_bijection(x['observation'], y['observation'], forward, reverse): return False
            if any(x['sample'].get(k, 0) != y['sample'].get(k, 0) for k in fields): return False
            if x['sent_command'] != y['sent_command']: return False
            if (f in left['rows']) != (f in right['rows']): return False
            if f in left['rows'] and left['rows'][f]['features'] != right['rows'][f]['features']: return False
        return before is not None or left['metrics'] == right['metrics']

    outcomes = []
    for ci in range(4):
        members = [d for d in loaded if d['variant']['context_index'] == ci]
        baseline = next(d for d in members if d['variant']['label'] == 'source' and d['variant']['repeat'] == 0)
        case = baseline['variant']['context']; frame = case['frame']
        ordinary = [d for d in members if d['variant']['label'] in ('source', 'identity')]
        exact_controls = all(equivalent(d, baseline) for d in ordinary)
        for d in members:
            v = d['variant']; prefix = equivalent(d, baseline, frame)
            at = next(s for s in d['steps'] if s['observation']['identity']['frame'] == frame)
            base_at = next(s for s in baseline['steps'] if s['observation']['identity']['frame'] == frame)
            features = d['rows'][frame]['features']; same_state = physical_equal_with_entity_bijection(
                at['observation'], base_at['observation'], {}, {}) and features == baseline['rows'][frame]['features']
            selected_hash = hashlib.sha256(json.dumps(features, separators=(',', ':')).encode()).hexdigest() == case['features_sha256']
            cold = next(q for q in members if q['variant']['label'] == v['label'] and q['variant']['repeat'] != v['repeat'])
            cold_exact = equivalent(d, cold)
            unchanged_draws = True; native_request = None; equipped = None; observed_ready = None
            log = (d['native_root']/'server.log').read_text(encoding='utf-8-sig')
            if v['label'] == 'shotgun':
                identity = next(q for q in members if q['variant']['label'] == 'identity' and q['variant']['repeat'] == v['repeat'])
                ir = next(s for s in identity['steps'] if s['observation']['identity']['frame'] == frame)['sample']['weapon_probe']
                rr = at['sample']['weapon_probe']
                unchanged_draws = all(rr[k] == ir[k] for k in ('weapon_uniform', 'fire_uniform', 'gaussian_noise'))
                unchanged_draws &= all(at['sample'].get(k, 0) == base_at['sample'].get(k, 0) for k in ('target', 'aim_mode', 'vertical'))
                requests = [int(f) for f in re.findall(r'sv_test_client_cmd frame=(\d+) seq=\d+ state=3 command=use Shotgun\b', log)]
                native_request = next((f for f in requests if frame <= f <= frame+1), None)
                observed = [s['observation'] for s in d['all_steps'] if frame <= s['observation']['identity']['frame']
                    < d['result']['first_life']['end_frame'] and s['observation']['identity']['life'] == 1
                    and s['observation']['health'] > 0 and s['observation']['weapon'] == 'Shotgun']
                equipped = observed[0]['identity']['frame'] if observed else None
                # Conservative UDP-visible idle/ready evidence, not hidden state:
                # SG generic idle frames19..36, firing can still occupy19.
                observed_ready = next((o['identity']['frame'] for o in observed if 20 <= o['gun_frame'] <= 36), None)
            goal = d['result']['goal_stop']; end = goal['kill_frame'] if goal else d['result']['first_life']['end_frame']
            wanted_mod = 2 if v['label'] == 'shotgun' else 1
            shots = []
            for event in re.finditer(r'g_test_(?:hitscan|projectile) version=1 event=(?:fire|spawn) ([^\r\n]+)', log):
                f = dict(word.split('=', 1) for word in event.group(1).split())
                if int(f.get('actor', f.get('attacker', '-1'))) != at['observation']['identity']['actor']+1:
                    continue
                if int(f['mod']) == wanted_mod and frame <= int(f['frame']) <= end:
                    shots.append(int(f['frame']))
            rules_before_end = sum(s['owner'] == 'rules' and frame <= s['observation']['identity']['frame'] < end
                                   and s['observation']['health'] > 0 for s in d['all_steps'])
            admitted = exact_controls and prefix and same_state and selected_hash and cold_exact and unchanged_draws and rules_before_end == 0
            if v['label'] == 'shotgun': admitted &= native_request is not None and equipped is not None
            outcomes.append(dict(map=case['map'], seed=case['seed'], label=v['label'], repeat=v['repeat'],
                admitted=bool(admitted), exact_controls=exact_controls, prefix_exact=prefix, root_state_exact=same_state,
                selected_feature_hash_exact=selected_hash, cold_repeat_exact=cold_exact, draws_exact=unchanged_draws,
                native_request_frame=native_request, observed_equipped_frame=equipped,
                first_physical_shot_frame=min(shots) if shots else None,
                conservative_observed_ready_frame=observed_ready,
                previous_weapon_request=at['observation'].get('previous_weapon_request'),
                rules_commands_before_end=rules_before_end, **d['metrics']))
    a.out.mkdir(parents=True)
    report = dict(state='complete', version='combat_weapon_choice_native_acceptance_v1', results=outcomes,
        admitted=sum(r['admitted'] for r in outcomes), excluded=sum(not r['admitted'] for r in outcomes),
        training_performed=False, promotion=False, neural_execution=False,
        scope='Fixed TRAIN-root causal diagnostics only; excluded comparisons cannot support labels or weapon-effect claims.')
    save(a.out/'report.json', report); save(a.out/'complete.json', dict(state='complete', report_sha256=sha(a.out/'report.json')))
    print(dict(admitted=report['admitted'], excluded=report['excluded']))


if __name__ == '__main__':
    main()
