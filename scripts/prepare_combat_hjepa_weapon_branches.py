"""Freeze diagnostic constant weapon policies selected from initial availability."""
import argparse
import hashlib
import json
import pathlib

NAMES = ['', 'Blaster', 'Shotgun', 'Super Shotgun', 'Machinegun', 'Chaingun', 'Grenade Launcher',
         'Rocket Launcher', 'HyperBlaster', 'Railgun', 'BFG10K', 'Grenades']
AMMO = ['', '', 'Shells', 'Shells', 'Bullets', 'Bullets', 'Grenades', 'Rockets', 'Cells', 'Slugs', 'Cells', 'Grenades']
COST = [0, 0, 1, 2, 1, 1, 1, 1, 1, 1, 50, 1]


def choices(instance):
    inv = instance['inventory']
    stock = {i['name']: i['count'] for i in inv['items']}
    return [i for i in range(1, 12) if NAMES[i] != inv['initial_weapon'] and stock.get(NAMES[i], 0) > 0
            and (not AMMO[i] or stock.get(AMMO[i], 0) >= COST[i])]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=pathlib.Path, required=True)
    ap.add_argument('--parent', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    a = ap.parse_args()
    plan = json.loads(a.plan.read_text(encoding='utf-8-sig'))
    if any(t['split'] not in ['train', 'validation'] for t in plan['tasks']):
        raise ValueError('training branches cannot use test/confirmation')
    instances = [i for t in plan['tasks'] for i in t['instances']]
    parent = json.loads(a.parent.read_text())
    if parent['kind'] != 'combat_hjepa_v1' or parent.get('weapon_mode') != 'keep_equipped_v1':
        raise ValueError('frozen keep-equipped H-JEPA required')
    a.out.mkdir(parents=True, exist_ok=False)
    variants = []
    for slot in range(12):
        seeds = [i['engine_seed'] for i in instances if slot == 0 or slot in choices(i)]
        if not seeds:
            continue
        model = dict(parent)
        if slot:
            bias = [0.] * 12
            bias[slot] = 50.
            model.update(weapon_mode='learned_masked_v1', weapon_head='combat_masked_weapon_v1',
                         weapon_actor=[dict(weight=[[0.] * 845 for _ in range(12)], bias=bias)])
            model.pop('weapon_sampling_seed', None)
        name = 'diagnostic-' + str(slot)
        p = a.out / (name + '.json')
        p.write_text(json.dumps(model, separators=(',', ':')), encoding='utf-8')
        variants.append(dict(name=name, mode='learned', model=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                             contract='stock_weapons_fire_v1', seeds=seeds, diagnostic_weapon_slot=slot))
    (a.out / 'models.json').write_text(json.dumps(variants, indent=2))
    (a.out / 'protocol.json').write_text(json.dumps(dict(version='hjepa_native_weapon_branches_v1', diagnostic=True,
        promotion=False, parent=str(a.parent.resolve()), parent_sha256=hashlib.sha256(a.parent.read_bytes()).hexdigest(),
        plan=str(a.plan.resolve()), plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
        branches=sum(len(v['seeds']) for v in variants), initial_states=len(instances),
        scope='Constant diagnostic weapon policies until availability is lost, then keep. Same motor/world. '
              'Available options selected before outcomes. Whole-policy contrasts, not optimal per-frame labels.'), indent=2))
    print(json.dumps({v['name']: len(v['seeds']) for v in variants}))


if __name__ == '__main__':
    main()
