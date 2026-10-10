"""Verify seeded peer walking, ordinary observations and closed native receipts."""
import argparse
import json
import math
import re
from pathlib import Path

from audit_coop_paired_export import read, sha, lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    report = read(root / 'report.json')
    assert report['accepted'] and report['moving_peer'] and report['source_unchanged'] and report['native_source_unchanged']
    receipt = read(root / 'command-receipt-audit.json')
    assert receipt['all_command_fields_exact']
    assert receipt['completed_pairs'] == report['completed_pairs']
    assert receipt['commands_matched'] == report['applied_commands']
    for path, digest in receipt['source_sha256'].items():
        assert sha(Path(path)) == digest
    log = (root / 'server.log').read_text(encoding='utf-8-sig')
    release = re.findall(r'sv_test_combat spawncount=\d+ server_frame=(\d+) g_test_combat_start game_frame=100 ready=2', log)
    assert len(release) == 1
    frame = int(release[0])
    peer = list(lines(root / 'PairLeader.jsonl'))
    peer = [row for row in peer if row.get('frame', -1) >= frame]
    assert peer
    moving = [row for row in peer if row['sent_command']['Forward'] or row['sent_command']['Side'] or row['sent_command']['Up']]
    assert moving and min(row['frame'] for row in moving) >= frame + 25
    assert max(row['frame'] for row in moving) < frame + 105
    assert all(not row['sent_command']['Buttons'] & 1 for row in peer)
    start = [128, 32, 24.125]
    maximum = max(math.dist(row['self'], start) for row in peer)
    target = [float(v) for v in report['peer_route']['target'].split(',')]
    final_error = math.dist(peer[-1]['self'], target)
    dead = [row for row in peer if row['health'] <= 0]
    if dead:
        assert maximum > 16
        assert re.search(r'g_test_damage .*target=2 .*health_after=0 .*target_class=player', log)
        assert all(not row['sent_command']['Forward'] and not row['sent_command']['Side'] for row in dead)
    else:
        assert maximum > 64 and final_error < 24
    assert all(math.dist(row['self'], start) < 1 for row in peer if row['frame'] <= frame + 25)
    learner = list(lines(root / 'PairLearner.jsonl'))
    controlled = [row for row in learner if row.get('combat_policy', {}).get('selection', {}).get('owner') == 'provider']
    assert controlled
    for row in controlled:
        observation = row['combat_policy']['observation']
        assert observation.get('navigation_context') is not None
        relative = observation.get('teammate_relative')
        if relative is not None:
            assert row.get('teammate') is not None
            expected = [mate - own for mate, own in zip(row['teammate'], observation['position'])]
            assert all(abs(a-b) < 1e-6 for a, b in zip(relative, expected))
    result = dict(version='coop_moving_peer_audit_v1', seed=report['seed'],
                  pairs=report['completed_pairs'], release_server_frame=frame,
                  first_walk_frame=min(row['frame'] for row in moving),
                  maximum_peer_displacement=maximum, final_target_distance=final_error,
                  route_reached=not dead and final_error < 24,
                  peer_died=bool(dead), first_observed_peer_death_frame=dead[0]['frame'] if dead else None,
                  provider_steps=len(controlled), peer_attacks=0,
                  ppo_trainable=False, training_performed=False,
                  source_sha256={str(root/name): sha(root/name) for name in
                                 ('report.json', 'PairLeader.jsonl', 'PairLearner.jsonl', 'server.log', 'command-receipt-audit.json')})
    (root / 'moving-peer-verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
