"""Scalar physical-shot witnesses with explicit server/game/observation clock domains."""
import re
from combat_equipped_weapon_features import equipped_weapon_index


def physical_shots(steps, log, start_server_frame, end_server_frame):
    actor = steps[0]['observation']['identity']['actor']
    shots = []
    for line in log.splitlines():
        match = re.search(r'sv_test_projectile spawncount=\d+ server_frame=(\d+) g_test_projectile version=1 event=spawn (.+)', line)
        if not match: continue
        server_frame = int(match[1]); f = dict(word.split('=', 1) for word in match[2].split())
        if int(f.get('actor', f.get('attacker', '-1'))) != actor or f.get('mod') != '1': continue
        if start_server_frame <= server_frame < end_server_frame:
            shots.append(dict(server_command_frame=server_frame, native_game_frame=int(f['frame']),
                              weapon='Blaster', evidence='native_projectile_spawn'))
    for left, right in zip(steps, steps[1:]):
        before, after = left['observation'], right['observation']; native = left.get('native_step') or {}
        execution = left.get('server_execution') or {}; frame = before['identity']['frame']
        if not (native.get('begin_frame') == frame and native.get('end_frame') == after['identity']['frame'] == frame+1
                and execution.get('matched') and execution.get('window_exclusive') and execution.get('recovery_commands') == 0): continue
        if not (before['identity']['life'] == after['identity']['life'] == 1 and before['health'] > 0): continue
        if (all(equipped_weapon_index(o) == 2 for o in (before, after)) and after['ammo'] < before['ammo']
                and after['gun_frame'] == 9 and start_server_frame <= frame < end_server_frame):
            shots.append(dict(server_command_frame=frame, observation_after_server_frame=after['identity']['frame'],
                              weapon='Shotgun', evidence='observed_shell_consumption_with_exclusive_native_step'))
    return sorted(shots, key=lambda x: x['server_command_frame'])
