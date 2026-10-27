"""Native shot/contact audit; observed-body geometry is diagnostic, not a mask."""
import argparse
import collections
import json
import math
import shutil
from pathlib import Path

from process_combat_architecture_pool import read, save, sha
from report_combat_machinegun_hits import scan, vector
from report_registered_machinegun_outcomes import outcome, summary
from report_combat_machinegun_aim import measure, summary as aim_summary


def box_entry(origin, direction, lower, upper):
    near, far = 0., 8192.
    for p, d, lo, hi in zip(origin, direction, lower, upper):
        if abs(d) < 1e-12:
            if not lo <= p <= hi:
                return None
        else:
            a, b = sorted(((lo-p)/d, (hi-p)/d))
            near, far = max(near, a), min(far, b)
            if far < near:
                return None
    return near


def self_test():
    assert box_entry((0,0,0),(1,0,0),(2,-1,-1),(4,1,1)) == 2
    assert box_entry((0,2,0),(1,0,0),(2,-1,-1),(4,1,1)) is None
    assert box_entry((0,0,0),(-1,0,0),(2,-1,-1),(4,1,1)) is None
    assert box_entry((3,0,0),(0,1,0),(2,-1,-1),(4,1,1)) == 0


def observed_contact(shot, step):
    o, action = step['observation'], step['action']
    body_set = o.get('observed_dynamic_bodies')
    if body_set is None:
        return dict(body_set_available=False)
    assert body_set['frame'] == o['identity']['frame']
    bodies = body_set['bodies']
    native_id = int(shot['fields']['target'])
    actual_body = next((b for b in bodies if b['id'] == native_id), None)
    entity = action.get('target_entity', 0)
    target = next((e for e in o['enemies'] if e['id'] == entity and e.get('observed_track') == action.get('target_track')), None)
    result = dict(body_set_available=True, actual_contact_body_observed=actual_body is not None,
        actual_contact_observed_dead=bool(actual_body and actual_body['observed_death_animation']),
        actual_contact_is_selected=native_id == entity, selected_target_available=target is not None,
        corpse_between_selected_center=None)
    if target is None or not target.get('clear_shot'):
        return result
    solid = target.get('observed_solid', 0)
    if solid in (0,31):
        return result
    bottom, top = -((solid >> 5)&31)*8, ((solid >> 10)&63)*8-32
    height = (top+bottom)/2 if top-bottom < 16 else min(max(22,bottom+8),top-8)
    # This geometry deliberately uses the actual native muzzle only as a
    # diagnostic reference. It is NOT a causal policy feature or training input.
    origin = vector(shot['fields']['start'])
    point = [o['position'][i]+target['relative'][i] for i in range(3)]
    point[2] += height
    length = math.dist(origin, point)
    if length < 1e-9:
        return result
    direction = [(point[i]-origin[i])/length for i in range(3)]
    width = (solid&31)*8
    target_origin = [o['position'][i]+target['relative'][i] for i in range(3)]
    target_entry = box_entry(origin, direction,
        [target_origin[0]-width,target_origin[1]-width,target_origin[2]+bottom],
        [target_origin[0]+width,target_origin[1]+width,target_origin[2]+top])
    assert target_entry is not None
    blocked = []
    for body in bodies:
        if body['id'] == entity or not body['protocol_bounds_known'] or not body['observed_death_animation']:
            continue
        center = [o['position'][i]+body['relative'][i] for i in range(3)]
        distance = box_entry(origin, direction,
            [center[i]+body['protocol_mins'][i] for i in range(3)],
            [center[i]+body['protocol_maxs'][i] for i in range(3)])
        if distance is not None and distance < target_entry:
            blocked.append(body['id'])
    result['corpse_between_selected_center'] = bool(blocked)
    result['selected_center_blockers'] = blocked
    return result


def run(root, out):
    assert not out.exists()
    self_test()
    seal = read(root/'analysis/complete.json')
    quality = read(root/'analysis/report.json')
    assert seal['state'] == quality['state'] == 'complete' and sha(root/'analysis/report.json') == seal['report_sha256']
    pool = read(root/'report.json')
    assert pool['state'] == 'complete' and pool['source_unchanged']
    pins = {str(root/'analysis/report.json'):sha(root/'analysis/report.json')}
    for name in ('audit_combat_observed_corpse_contacts.py','report_combat_machinegun_hits.py',
                 'report_registered_machinegun_outcomes.py','report_combat_machinegun_aim.py','native_projectile_window.py'):
        path = Path(__file__).with_name(name).resolve()
        pins[str(path)] = sha(path)
    out.mkdir(parents=True)
    shutil.copyfile(__file__,out/Path(__file__).name)
    cases = []
    for job in sorted(pool['jobs'],key=lambda j:j['job']):
        if job['label'] == 'rules':
            continue
        capture = read(Path(job['root'])/'report.json')
        assert capture['capture_complete'] and capture['provenance_valid']
        directory = Path(capture['results'][0]['root'])
        paths = {k:directory/v for k,v in dict(steps='dataset/steps.jsonl', effects='dataset/server_outcomes.jsonl', log='server.log').items()}
        for path in paths.values():
            pins[str(path)] = sha(path)
        steps = [json.loads(line) for line in paths['steps'].open(encoding='utf-8-sig')]
        windows, excluded_dispatch = {}, 0
        for s in steps:
            n, execution, o = s.get('native_step'), s.get('server_execution') or {}, s['observation']
            if not n or s['owner'] != 'provider' or o['identity']['life'] != 1 or o['health'] <= 0 or o['identity']['frame'] < 100:
                continue
            if not execution.get('matched') or not execution.get('window_exclusive') or execution.get('recovery_commands') != 0:
                excluded_dispatch += 1
                continue
            key = tuple(n[k] for k in ('spawncount','begin_frame','actor','sequence'))
            assert key not in windows
            windows[key] = s
        shots = [q for q in scan(paths['log'].open(encoding='utf-8-sig'),set(windows)) if q['eligible']]
        records = []
        for shot in shots:
            step = windows[shot['window']]
            records.append(dict(frame=shot['window'][1], outcome=outcome(shot),
                contact_entity=int(shot['fields']['target']), selected_entity=step['action'].get('target_entity',0),
                observation=observed_contact(shot,step), aim=measure(shot,step)))
        phases = {}
        for phase, predicate in (('all',lambda f:True),('after320',lambda f:f>=320)):
            selected = [q for q in shots if predicate(q['window'][1])]
            matching = [r for r in records if predicate(r['frame'])]
            selected_steps = [s for s in windows.values() if predicate(s['observation']['identity']['frame'])]
            corpses = [r for r in matching if r['outcome'] == 'corpse_contact']
            phases[phase] = dict(machinegun=summary(selected),
                aim=aim_summary([r['aim'] for r in matching if r['aim']['status'] == 'measured']),
                body_flags=dict(collections.Counter(k for r in corpses for k,v in r['observation'].items() if v is True)),
                corpse_contacts_selected_same=sum(r['contact_entity']==r['selected_entity'] for r in corpses),
                actor_rows=len(selected_steps), attack_sent_rows=sum(bool(s['sent_command']['Buttons']&1) for s in selected_steps),
                weapon_rows=dict(collections.Counter(s['observation']['weapon'] for s in selected_steps)))
        cases.append(dict(map=job['case_name'],label=job['label'],seed=job['seed'],phases=phases,shots=records,excluded_dispatch_rows=excluded_dispatch,
            ending_inventory=steps[-1]['observation']['inventory']))
    totals = {}
    for label in ('control','request'):
        chosen = [c for c in cases if c['label']==label]
        shots = [r for c in chosen for r in c['shots']]
        corpses = [r for r in shots if r['outcome']=='corpse_contact']
        totals[label] = dict(shots=len(shots),outcomes=dict(collections.Counter(r['outcome'] for r in shots)),
            corpse_cases=sum(any(r['outcome']=='corpse_contact' for r in c['shots']) for c in chosen),
            corpse_contact_observed_dead=sum(r['observation'].get('actual_contact_observed_dead',False) for r in corpses),
            corpse_contact_selected_same=sum(r['contact_entity']==r['selected_entity'] for r in corpses),
            selected_center_blocked_for_corpse_contact=sum(r['observation'].get('corpse_between_selected_center') is True for r in corpses))
    assert all(sha(Path(p)) == digest for p,digest in pins.items())
    save(out/'report.json',dict(state='complete',totals=totals,cases=cases,inputs=pins,training_performed=False,
        scope='Existing development worlds; actual matched native MG bullets only. No shots inferred from attack commands. Geometry uses native muzzle as diagnostic reference, not causal policy input, ignores BSP and moving-body uncertainty. No actor change or runtime fire mask.'))
    save(out/'complete.json',dict(state='complete',report_sha256=sha(out/'report.json')))
    print(json.dumps(totals,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    run(args.root.resolve(),args.out.resolve())
