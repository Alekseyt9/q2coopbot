"""Audit learned choices against observed inventory and subsequent weapon models."""
import argparse
import collections
import json
import pathlib

NAMES = ['', 'Blaster', 'Shotgun', 'Super Shotgun', 'Machinegun', 'Chaingun',
         'Grenade Launcher', 'Rocket Launcher', 'HyperBlaster', 'Railgun', 'BFG10K', 'Grenades']
MODELS = ['v_blast', 'v_shotg', 'v_shotg2', 'v_machn', 'v_chain', 'v_launch',
          'v_rocket', 'v_hyperb', 'v_rail', 'v_bfg', 'v_handgr']
AMMO = ['', '', 'Shells', 'Shells', 'Bullets', 'Bullets', 'Grenades', 'Rockets', 'Cells', 'Slugs', 'Cells', 'Grenades']
MINIMUM = [0, 0, 1, 2, 1, 1, 1, 1, 1, 1, 50, 1]


def canonical(name):
    for weapon, model in zip(NAMES[1:], MODELS):
        if name in [weapon, 'models/weapons/' + model + '/tris.md2']:
            return weapon
    return None


def available(o, name):
    if name == '':
        return True
    age = o.get('inventory_age_frames')
    if o.get('inventory') is None or age is None or not 0 <= age <= 20:
        return False
    items = {i['name']: i['count'] for i in o['inventory']}
    i = NAMES.index(name)
    return items.get(name, 0) > 0 and (not AMMO[i] or items.get(AMMO[i], 0) >= MINIMUM[i])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    a = ap.parse_args()
    counts = collections.Counter()
    counts['unavailable_choices'] = 0
    requests = collections.Counter()
    switches = []
    for path in a.root.glob('case-*/worker-*/dataset/steps.jsonl'):
        rows = [json.loads(line) for line in path.open(encoding='utf-8')]
        pending = []
        previous = None
        for s in rows:
            o = s['observation']
            if not s.get('native_step') or o['health'] <= 0:
                continue
            if not s['server_execution']['matched']:
                raise ValueError('unverified execution')
            if not s['server_execution']['window_exclusive']:
                counts['nonexclusive_terminal_windows_omitted'] += 1
                continue
            frame = o['identity']['frame']
            life = (o['identity']['life'], o['identity']['spawncount'], o['identity']['map'])
            current = canonical(o['weapon'])
            if previous and previous[0] == life and current and previous[1] and current != previous[1]:
                matching = [(f, w) for f, w, ident in pending if ident == life and w == current and 0 < frame - f <= 30]
                counts['observed_changes'] += 1
                if matching:
                    counts['observed_after_provider_request'] += 1
                switches.append(dict(source=str(path), frame=frame, before=previous[1], after=current,
                                     provider_requests_in_previous_30_frames=matching))
            if current:
                previous = (life, current)
            if s['owner'] != 'provider':
                counts['nonprovider_native_frames'] += 1
                continue
            counts['provider_native_frames'] += 1
            weapon = s['action'].get('weapon', '')
            if not available(o, weapon):
                counts['unavailable_choices'] += 1
            if weapon:
                requests[weapon] += 1
                counts['nonkeep_choices'] += 1
                if weapon != current:
                    counts['requests_different_from_equipped'] += 1
                    pending.append((frame, weapon, life))
    if not counts['provider_native_frames']:
        raise ValueError('no verified provider frames')
    report = dict(version='hjepa_weapon_choice_audit_v1', counts=dict(counts), requests=dict(requests), switches=switches,
                  scope='Native-joined provider choices. Subsequent observed weapon changes within 30 frames '
                        'are temporal acknowledgements, not proof that engine autoswitch played no role. '
                        'Non-provider actions and empty-ammo engine changes are separate. Not shot accuracy.')
    a.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if counts['unavailable_choices']:
        raise ValueError('unavailable learned weapon choices')
    print(json.dumps(dict(counts=counts, requests=requests), indent=2))


if __name__ == '__main__':
    main()
