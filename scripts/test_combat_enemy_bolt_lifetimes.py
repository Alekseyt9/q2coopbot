import unittest

from combat_enemy_bolt_lifetimes import match_lifetime, parse_lifetimes


READY = ['g_test_enemy_bolt ready version=1', 'g_test_projectile ready version=1']


def event(stream, frame, serial, entity, phase, attacker=7):
    index = 'bolt' if stream == 'enemy_bolt' else 'shot'
    value = f'sv_test_{stream} spawncount=42 server_frame={frame} g_test_{stream} version=1 event={phase} map=base3 frame={frame+900} {index}={serial} entity={entity}'
    if phase == 'spawn':
        value += f' attacker={attacker}'
        if stream == 'enemy_bolt':
            value += ' attacker_class=monster_soldier_light'
        value += ' x=0 y=0 z=20 vx=600 vy=0 vz=0'
    return value


class LifetimeTests(unittest.TestCase):
    def test_network_clock_and_entity_reuse(self):
        records = parse_lifetimes(READY + [event('projectile', 2, 1, 300, 'spawn', 1),
            event('projectile', 4, 1, 300, 'end'), event('enemy_bolt', 6, 1, 300, 'spawn'),
            event('enemy_bolt', 10, 1, 300, 'end')])
        identity = dict(map='base3', spawncount=42, frame=3, actor=1)
        self.assertEqual(match_lifetime(records, identity, 300)[1], 'self')
        identity['frame'] = 7
        record, label = match_lifetime(records, identity, 300)
        self.assertEqual((label, record['attacker_class']), ('enemy', 'monster_soldier_light'))
        for frame in (1, 4, 5, 10, 907):
            identity['frame'] = frame
            self.assertEqual(match_lifetime(records, identity, 300)[1], 'unmatched')
        identity.update(frame=7, spawncount=43)
        self.assertEqual(match_lifetime(records, identity, 300)[1], 'unmatched')

    def test_same_frame_reuse_stays_ambiguous(self):
        records = parse_lifetimes(READY + [event('projectile', 2, 1, 300, 'spawn', 1),
            event('projectile', 4, 1, 300, 'end'), event('enemy_bolt', 4, 1, 300, 'spawn')])
        self.assertEqual(match_lifetime(records, dict(map='base3', spawncount=42, frame=4, actor=1), 300)[1], 'same_frame_reuse')

    def test_malformed_and_unbound_events_rejected(self):
        variants = [READY + [event('enemy_bolt', 1, 1, 300, 'end')],
            READY + [event('enemy_bolt', 1, 1, 300, 'spawn'), event('projectile', 2, 1, 300, 'spawn')],
            [event('enemy_bolt', 1, 1, 300, 'spawn')],
            READY + [event('enemy_bolt', 1, 1, 300, 'spawn').replace('vx=600', 'vx=nan')]]
        for lines in variants:
            with self.subTest(lines=lines), self.assertRaises(AssertionError):
                parse_lifetimes(lines)


if __name__ == '__main__':
    unittest.main()
