"""Strict offline SG trigger/pellet/pose stream; never policy inputs or rewards."""
import argparse
import json
import math
import re
from pathlib import Path
from evaluate_registered_combat_models import save, sha

SG = re.compile(r'^sv_test_sg spawncount=(-?\d+) server_frame=(\d+) g_test_sg version=1 (.*)$')
STEP = re.compile(r'^sv_test_step version=1 phase=(begin|end) spawncount=(-?\d+) frame=(\d+) seq=(\d+) actor=(\d+)$')
DAMAGE = re.compile(r'^sv_test_damage spawncount=(-?\d+) server_frame=(\d+) g_test_damage version=1 (.*)$')


def fields(text):
    result = {}
    for token in text.split():
        key, value = token.split('=', 1)
        if key in result:
            raise ValueError('Duplicate telemetry field: '+key)
        result[key] = value
    return result


def vector(text):
    values = [float(v) for v in text.split(',')]
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError('Invalid telemetry vector')
    return values


def parse(lines):
    window, current, completed, seen = None, None, [], set()
    for raw in lines:
        line = raw.rstrip('\r\n')
        if line.startswith('sv_test_step '):
            m = STEP.fullmatch(line)
            if not m:
                raise ValueError('Malformed native step')
            phase, generation, frame, sequence, actor = m.groups()
            generation, frame, sequence, actor = map(int, (generation, frame, sequence, actor))
            if phase == 'begin':
                if window is not None or current is not None:
                    raise ValueError('Nested native window')
                window = dict(spawncount=generation, begin_frame=frame, sequence=sequence, actor=actor)
            else:
                if current is not None or window is None or any(window[k] != v for k, v in [('spawncount', generation), ('sequence', sequence), ('actor', actor)]):
                    raise ValueError('Unclosed trigger or unmatched native window')
                if frame != window['begin_frame']+1:
                    raise ValueError('Noncontiguous native window')
                window = None
            continue
        if line.startswith('sv_test_sg '):
            m = SG.fullmatch(line)
            if m is None or window is None:
                raise ValueError('Shotgun telemetry outside a native window')
            generation, server_frame = map(int, m.groups()[:2]); f = fields(m[3])
            required = {
                'begin': 'mod count damage start aim origin velocity view viewheight grounded ducked gunframe ammo_before',
                'pose': 'entity class health solid origin mins maxs velocity',
                'pellet': 'mod index count start aim spread nominal water muzzle_blocked fraction impact sky target target_class damageable health_before damage',
                'end': 'mod count',
            }
            if f.get('event') not in required or not set(('event map frame trigger actor '+required[f['event']]).split()) <= f.keys():
                raise ValueError('Missing or unknown Shotgun fields')
            # A ready weapon may fire in ClientThink at the beginning or in
            # ClientBeginServerFrame at the end, before this window is closed.
            if generation != window['spawncount'] or server_frame not in (window['begin_frame'], window['begin_frame']+1) or int(f['actor']) != window['actor']:
                raise ValueError('Shotgun/native identity differs')
            key = (generation, int(f['trigger']))
            if f['event'] == 'begin':
                if current is not None or key in seen or not 1 <= int(f['count']) <= 64 or int(f['mod']) not in (2, 3):
                    raise ValueError('Invalid/duplicate trigger')
                seen.add(key)
                current = dict(spawncount=generation, server_frame=server_frame, window=dict(window),
                               native_phase_offset=server_frame-window['begin_frame'],
                               trigger=key[1], begin=f, poses=[], pellets=[], damage=[])
                for name in ('start', 'aim', 'origin', 'velocity', 'view'):
                    vector(f[name])
                if not math.isfinite(float(f['viewheight'])) or any(f[k] not in ('0', '1') for k in ('grounded', 'ducked')):
                    raise ValueError('Invalid firing stance')
            elif current is None or current['trigger'] != key[1] or server_frame != current['server_frame'] or f['map'] != current['begin']['map'] or f['frame'] != current['begin']['frame']:
                raise ValueError('Trigger identity differs')
            elif f['event'] == 'pose':
                if current['pellets'] or any(p['entity'] == f['entity'] for p in current['poses']):
                    raise ValueError('Late or duplicate pose')
                for name in ('origin', 'mins', 'maxs', 'velocity'):
                    vector(f[name])
                current['poses'].append(f)
            elif f['event'] == 'pellet':
                if int(f['index']) != len(current['pellets']) or f['count'] != current['begin']['count'] or f['mod'] != current['begin']['mod']:
                    raise ValueError('Pellet sequence/count differs')
                if len(current['pellets']) >= int(f['count']):
                    raise ValueError('Too many pellets')
                for name in ('start', 'aim', 'impact'):
                    vector(f[name])
                if not math.isfinite(float(f['fraction'])) or not 0 <= float(f['fraction']) <= 1:
                    raise ValueError('Invalid trace fraction')
                spread = [float(v) for v in f['spread'].split(',')]
                if len(spread) != 2 or not all(math.isfinite(v) for v in spread):
                    raise ValueError('Invalid spread')
                current['pellets'].append(f)
            elif f['event'] == 'end':
                if f['count'] != current['begin']['count'] or f['mod'] != current['begin']['mod'] or len(current['pellets']) != int(f['count']):
                    raise ValueError('Incomplete trigger')
                current['direct_monster_damage'] = sum(max(0, min(int(d['health_before']), int(d['health_before'])-int(d['health_after'])))
                    for d in current['damage'] if d.get('target_class', '').startswith('monster_')
                    and d['attacker'] == current['begin']['actor'] and d['mod'] == current['begin']['mod'])
                completed.append(current); current = None
            else:
                raise ValueError('Unknown Shotgun event')
            continue
        if line.startswith('g_test_sg '):
            raise ValueError('Unwrapped Shotgun telemetry has no native identity')
        if current is not None and line.startswith('sv_test_damage '):
            m = DAMAGE.fullmatch(line)
            if m is None or int(m[1]) != current['spawncount'] or int(m[2]) != current['server_frame'] or not current['pellets']:
                raise ValueError('Damage cannot be associated with a pellet')
            d = fields(m[3]); d['pellet_index'] = len(current['pellets'])-1
            current['damage'].append(d)
    if current is not None or window is not None:
        raise ValueError('Truncated native/Shotgun stream')
    return completed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--log', required=True, type=Path); ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--allow-empty', action='store_true')
    a = ap.parse_args(); assert not a.out.exists()
    with a.log.open(encoding='utf-8-sig') as f:
        rows = parse(f)
    assert rows or a.allow_empty, 'No native Shotgun triggers'
    save(a.out, dict(version='combat_native_shotgun_ballistics_v1', state='passed', triggers=rows,
         trigger_count=len(rows), pellet_count=sum(len(r['pellets']) for r in rows),
         direct_monster_damage=sum(r['direct_monster_damage'] for r in rows),
         neural_execution=False, policy_inputs=False, reward_integration=False,
         input=str(a.log.resolve()), input_sha256=sha(a.log),
         parser_sha256=sha(Path(__file__)),
         scope='Offline authoritative shot-time trigger/pellets/monster poses. Includes hidden poses; never supply these labels as runtime observations. Damage association is temporal within a pellet callback; direct damage additionally filters actor/mod.'))
    print(dict(triggers=len(rows), pellets=sum(len(r['pellets']) for r in rows)), flush=True)


if __name__ == '__main__':
    main()
