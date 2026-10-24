"""Metadata admission for training and inference environment execution semantics."""
LEGACY = 'legacy_v1'
CHECKED_GROUND = 'checked_ground_crouch_v1'
SCOPE = 'combat_execution_contract_ppo_v1'


def normalize(value):
    if value is None or value == '':
        return LEGACY
    if value not in (LEGACY, CHECKED_GROUND):
        raise ValueError('Unknown combat execution contract: '+repr(value))
    return value


def require_same(*values):
    contracts = [normalize(v) for v in values]
    if not contracts or len(set(contracts)) != 1:
        raise ValueError('Mixed combat execution contracts: '+repr(contracts))
    return contracts[0]
