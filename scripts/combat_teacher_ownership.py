"""Distinguish active rules control from a proven passive goal-stop receipt."""


def passive_goal_stop(step, goal):
    if not goal or goal.get('version') != 'combat_goal_stop_v2':
        return False
    objective = goal.get('native_objective') or {}
    current = step.get('observation', {}).get('identity', {})
    following = step.get('next_observation') or {}
    nxt = following.get('identity', {})
    native = step.get('native_step') or {}
    dispatch = step.get('server_execution') or {}
    sent = step.get('sent_command') or {}
    frame = current.get('frame')
    return bool(
        step.get('owner') == 'rules' and step.get('terminal') is True
        and not step.get('truncated') and step.get('reason') == 'combat_goal_complete'
        and goal.get('reason') == 'combat_goal_complete' and goal.get('health', 0) > 0
        and objective.get('alive') == objective.get('pending') == 0
        and type(frame) is int and frame == goal.get('kill_frame') and goal.get('observed_frame') == frame+1
        and objective.get('frame') == nxt.get('frame') == goal['observed_frame']
        and current.get('map') == nxt.get('map') == objective.get('map')
        and current.get('spawncount') == nxt.get('spawncount') == objective.get('spawncount') == goal.get('spawncount')
        and current.get('actor') == nxt.get('actor') == native.get('actor') == goal.get('actor')
        and current.get('life') == nxt.get('life') == 1 and following.get('health', 0) > 0
        and native.get('spawncount') == goal.get('spawncount')
        and native.get('begin_frame') == frame and native.get('end_frame') == nxt.get('frame')
        and native.get('damage_indexes') == []
        and dispatch.get('matched') is True and dispatch.get('window_exclusive') is True
        and dispatch.get('dispatch_frame') == frame and dispatch.get('recovery_commands') == 0
        and all(sent.get(key) == 0 for key in ('Forward','Side','Up','Buttons','Impulse')))


def rules_ownership(rows, goal, end_frame):
    limit = goal['kill_frame'] if goal else end_frame
    combat, passive, total = 0, [], 0
    for step in rows:
        if step['owner'] != 'rules' or step['observation']['identity']['frame'] > limit:
            continue
        total += 1
        if passive_goal_stop(step, goal):
            passive.append(step['observation']['identity']['frame'])
        else:
            combat += 1
    return dict(rules_combat_rows=combat, passive_goal_stop_frames=passive,
                original_inclusive_rules_rows=total)
