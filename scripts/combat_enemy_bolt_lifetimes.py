"""Offline native lifetime joins. These labels must never become policy inputs."""
import math
import re


PATTERN = re.compile(r'^sv_test_(enemy_bolt|projectile) spawncount=(-?\d+) server_frame=(\d+) g_test_(enemy_bolt|projectile) (.*)$')


def parse_lifetimes(lines):
    ready = set()
    records, active, generations = {}, {}, {}
    for line in lines:
        line = line.strip()
        if line in ('g_test_enemy_bolt ready version=1', 'g_test_projectile ready version=1'):
            ready.add('enemy_bolt' if line.startswith('g_test_enemy_bolt ') else 'projectile')
            continue
        if not line.startswith(('sv_test_enemy_bolt ', 'sv_test_projectile ', 'g_test_enemy_bolt ', 'g_test_projectile ')):
            continue
        match = PATTERN.fullmatch(line)
        assert match and match[1] == match[4], 'Malformed or unbound lifetime'
        stream, generation, frame = match[1], int(match[2]), int(match[3])
        assert stream in ready, 'Unready lifetime stream'
        items = [part.split('=', 1) for part in match[5].split()]
        assert all(len(part) == 2 for part in items) and len({part[0] for part in items}) == len(items)
        fields = dict(items)
        assert fields['version'] == '1' and fields['event'] in ('spawn', 'end')
        serial = int(fields['bolt' if stream == 'enemy_bolt' else 'shot'])
        entity, game_frame = int(fields['entity']), int(fields['frame'])
        assert serial > 0 and entity > 0 and frame >= 0 and game_frame >= 0
        map_name = fields['map']
        assert generations.setdefault(generation, map_name) == map_name, 'Generation map changed'
        token, slot = (stream, generation, serial), (generation, entity)
        if fields['event'] == 'spawn':
            assert token not in records and slot not in active, 'Duplicate/overlapping lifetime'
            attacker = int(fields['attacker'])
            assert attacker > 0
            velocity = [float(fields[key]) for key in ('vx', 'vy', 'vz')]
            origin = [float(fields[key]) for key in ('x', 'y', 'z')]
            assert all(math.isfinite(x) for x in velocity + origin)
            attacker_class = fields.get('attacker_class')
            if stream == 'enemy_bolt':
                assert attacker_class and attacker_class.startswith('monster_')
            record = dict(stream=stream, spawncount=generation, serial=serial, entity=entity, map=map_name,
                start=frame, end=None, game_start=game_frame, attacker=attacker,
                attacker_class=attacker_class, native_initial_velocity=velocity, native_initial_origin=origin)
            records[token] = record
            active[slot] = token
        else:
            assert token in records and active.get(slot) == token, 'Unmatched end or reused entity'
            record = records[token]
            assert record['entity'] == entity and record['start'] <= frame
            record['end'], record['game_end'] = frame, game_frame
            del active[slot]
    assert ready == {'enemy_bolt', 'projectile'}, 'Lifetime readiness missing'
    return list(records.values())


def match_lifetime(records, identity, entity):
    relevant = [r for r in records if r['spawncount'] == identity['spawncount'] and r['entity'] == entity and r['map'] == identity['map']]
    frame = identity['frame']
    # Network frame alone cannot order disappearance and re-use inside one frame.
    if any(r['end'] == frame for r in relevant) and any(r['start'] == frame for r in relevant):
        return None, 'same_frame_reuse'
    matches = [r for r in relevant if r['start'] <= frame and (r['end'] is None or frame < r['end'])]
    if len(matches) != 1:
        return None, 'unmatched'
    record = matches[0]
    if record['stream'] == 'enemy_bolt':
        assert record['attacker'] != identity['actor'], 'Enemy lifetime belongs to player'
        return record, 'enemy'
    return record, 'self' if record['attacker'] == identity['actor'] else 'other_player'
