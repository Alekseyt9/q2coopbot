"""Scalar admission for starting-ready weapon experiments; no neural evaluation."""
import argparse
import json
from pathlib import Path
import re
from evaluate_registered_combat_models import read, save, sha
from combat_equipped_weapon_features import equipped_weapon_index
from audit_combat_joint_aim_grid import physical_equal_with_entity_bijection
from combat_native_shot_timing import physical_shots


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--probe', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); root = a.probe.resolve(); assert not a.out.exists()
    seal, spec, pool = [read(root/p) for p in ('complete.json', 'spec.json', 'pool/report.json')]
    assert seal['native_cases'] == seal['cuda_cases'] == 16 and len(spec['variants']) == 16
    assert sha(root/'spec.json') == seal['spec_sha256'] and sha(root/'pool/report.json') == seal['pool_sha256']
    assert pool['source_unchanged'] and sha(root/seal.get('processing_root', '.')/'cuda-batch/report.json') == seal['cuda_receipt_sha256']
    jobs = {j['plan_index']: j for j in pool['jobs']}; loaded = []
    def lines(path):
        with path.open(encoding='utf-8-sig') as stream: return [json.loads(s) for s in stream]
    def inventory(o): return sorted((x['id'], x['name'], x['count']) for x in o['inventory'])
    for v in spec['variants']:
        i = v['plan_index']; directory = root/seal.get('processing_root', '.')/f'case-{i:02d}'; capture = Path(jobs[i]['root'])
        meta, proof = read(directory/'cuda/report.json'), read(directory/'cuda/cuda-verification.json')
        assert proof['state'] == 'passed' and proof['device'] == 'cuda'
        assert sha(directory/'cuda/cuda-verification.json') == meta['cuda_verification_sha256']
        assert meta['model_sha256'] == spec['source_weights_sha256'] == sha(v['model'])
        path = next((directory/'native').glob('replay-*/steps.jsonl'))
        assert sha(path) == meta['source_sha256'][str(path.resolve())]
        all_steps = lines(path)
        controlled = [s for s in all_steps if s['owner'] == 'provider' and s.get('sample') and s.get('native_step')
            and s['observation']['identity']['life'] == 1 and s['observation']['health'] > 0
            and (s.get('server_execution') or {}).get('matched')
            and (s.get('server_execution') or {}).get('window_exclusive')
            and (s.get('server_execution') or {}).get('recovery_commands') == 0]
        assert controlled
        rollout = directory/'cuda/rollout.jsonl'; assert sha(rollout) == meta['rollout_sha256']
        rows = {r['frame']: r for r in lines(rollout)}
        report = read(capture/'report.json'); result = report['results'][0]
        assert report['capture_complete'] and report['provenance_valid'] and result['generated_start']['confirmed']
        life = result['first_life']; damage = life['damage']; goal = result['goal_stop']
        end = goal['kill_frame'] if goal else life['end_frame']
        rules = sum(s['owner'] == 'rules' and 98 <= s['observation']['identity']['frame'] < end
                    and s['observation']['health'] > 0 for s in all_steps)
        log = (Path(result['root'])/'server.log').read_text(encoding='utf-8-sig')
        weapon, idle, slot = ('Shotgun', 19, 2) if v['label'] == 'shotgun' else ('Blaster', 9, 1)
        ready = len(re.findall(r'^g_test_weapon_start game_frame=100 actor=1 weapon='+weapon+r' gunframe_before=\d+ gunframe_after='+str(idle)+r'$', log, re.M)) == 1
        first = controlled[0]['observation']
        equipment = first['identity']['frame'] == 98 and equipped_weapon_index(first) == slot and 0 <= first['inventory_age_frames'] <= 2
        stock = inventory(first) == [(7, 'Blaster', 1), (8, 'Shotgun', 1), (18, 'Shells', 20)]
        metrics = dict(goal=bool(goal), death=bool(result['death_stop']), end_frame=life['end_frame'],
            kills=sum(q['monster_kills'] for q in damage['by_mod']), dealt=sum(q['monster_health_damage'] for q in damage['by_mod']),
            received=damage['received_health_damage'])
        start = sorted((e['class'], e['health'], e['solid'], tuple(e['position'])) for e in result['generated_start']['entities'])
        # Native game frames and UDP server frames differ. Never subtract the
        # native game-frame field from an observation frame.
        shots = physical_shots(all_steps, log, 98, end)
        loaded.append(dict(variant=v, steps=controlled, rows=rows, metrics=metrics, start=start,
            ready=ready, equipment=equipment, stock=stock, rules=rules,
            first_shot=shots[0] if shots else None))
    fields = ('sampling_seed', 'latent', 'attack', 'vertical', 'weapon', 'target', 'aim_mode',
              'memory', 'aim_candidate', 'log_probability', 'value')
    def repeat_equal(a, b):
        if len(a['steps']) != len(b['steps']) or a['metrics'] != b['metrics']: return False
        forward, reverse = {}, {}
        for x, y in zip(a['steps'], b['steps']):
            f = x['observation']['identity']['frame']
            if f != y['observation']['identity']['frame']: return False
            if not physical_equal_with_entity_bijection(x['observation'], y['observation'], forward, reverse): return False
            if any(x['sample'].get(k, 0) != y['sample'].get(k, 0) for k in fields): return False
            if x['sent_command'] != y['sent_command']: return False
            if (f in a['rows']) != (f in b['rows']): return False
            if f in a['rows'] and a['rows'][f]['features'] != b['rows'][f]['features']: return False
        return True
    outcomes = []
    for d in loaded:
        v = d['variant']; members = [q for q in loaded if q['variant']['context_index'] == v['context_index']]
        counterpart = next(q for q in members if q['variant']['label'] != v['label'] and q['variant']['repeat'] == v['repeat'])
        twin = next(q for q in members if q['variant']['label'] == v['label'] and q['variant']['repeat'] != v['repeat'])
        same_start = d['start'] == counterpart['start']
        cold = repeat_equal(d, twin)
        admitted = d['ready'] and d['equipment'] and d['stock'] and counterpart['stock'] and same_start and cold and d['rules'] == 0
        outcomes.append(dict(map=v['context']['map'], seed=v['context']['seed'], label=v['label'], repeat=v['repeat'],
            admitted=admitted, native_ready_at_100=d['ready'], observed_initial_equipment=d['equipment'],
            inventory_exact=d['stock'], matched_nonweapon_native_start=same_start, cold_repeat_exact=cold,
            rules_commands_before_end=d['rules'], first_physical_shot=d['first_shot'], **d['metrics']))
    a.out.mkdir(parents=True)
    report = dict(state='complete', version='combat_ready_weapon_train_acceptance_v2_server_clock', results=outcomes,
        admitted=sum(r['admitted'] for r in outcomes), excluded=sum(not r['admitted'] for r in outcomes),
        independent_worlds=4, cold_repeats=2, training_performed=False, promotion=False, neural_execution=False,
        scope='Small TRAIN-only initial-state comparison; equipped weapon differs, subsequent actions remain policy controlled. No identical-state causal pairing, fitted labels, held-out quality or rules superiority claim.')
    save(a.out/'report.json', report)
    save(a.out/'complete.json', dict(state='complete', report_sha256=sha(a.out/'report.json')))
    print(dict(admitted=report['admitted'], excluded=report['excluded']))


if __name__ == '__main__': main()
