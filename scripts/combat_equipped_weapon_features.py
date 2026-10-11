"""Equipped weapon and own request history for the v10 observation contract.

Preserve the complete v9 prefix. Current weapon comes from observation;
previous request is local command history, never a server acknowledgement.
"""
import math

VERSION = 'combat_features_v10'
PREFIX = 1121
CATEGORIES = ('unknown', 'Blaster', 'Shotgun', 'Super Shotgun', 'Machinegun',
              'Chaingun', 'Grenade Launcher', 'Rocket Launcher', 'HyperBlaster',
              'Railgun', 'BFG10K', 'Grenades')
WIDTH = PREFIX + 26  # current12 + previous-request12 + age-known1 + age1
VIEW_MODELS = ('v_blast', 'v_shotg', 'v_shotg2', 'v_machn', 'v_chain',
               'v_launch', 'v_rocket', 'v_hyperb', 'v_rail', 'v_bfg', 'v_handgr')
ALIASES = {name.casefold(): index for index, name in enumerate(CATEGORIES[1:], 1)}
ALIASES.update({f'models/weapons/{model}/tris.md2': index
                for index, model in enumerate(VIEW_MODELS, 1)})


def equipped_weapon_index(observation):
    weapon = observation.get('weapon')
    if weapon is None:
        return 0
    if not isinstance(weapon, str):
        raise ValueError('Observed equipped weapon must be a string')
    return ALIASES.get(weapon.strip().replace('\\', '/').casefold(), 0)


def equipped_weapon_tail(observation):
    result = [0.] * len(CATEGORIES)
    result[equipped_weapon_index(observation)] = 1.
    return result


def append_equipped_weapon(prefix, observation):
    if len(prefix) != PREFIX or not all(isinstance(v, (int, float)) and
                                       not isinstance(v, bool) and math.isfinite(v) for v in prefix):
        raise ValueError('Expected finite v9 feature prefix of width 1121')
    return list(prefix) + equipped_weapon_tail(observation) + weapon_request_tail(observation)


def weapon_request_tail(observation):
    result = [0.] * 14
    result[0] = 1.
    request = observation.get('previous_weapon_request')
    age = observation.get('previous_weapon_request_age_frames')
    if request is None and age is None:
        return result
    if not isinstance(request,str) or not request or not isinstance(age,int) or isinstance(age,bool) or age < 0:
        raise ValueError('Incomplete or invalid observed own weapon request')
    index = equipped_weapon_index(dict(weapon=request))
    if not index:
        raise ValueError('Unknown own weapon request')
    result[0] = 0.
    result[index] = 1.
    result[12] = 1.
    result[13] = min(4.,age / 30)
    return result
