"""Metadata-only admission for checkpoints with dedicated optimizer layouts."""


DECISION_SCOPE = 'combat_decision_scope_ppo_v1'


def require_checkpoint_scope(checkpoint, expected=None):
    scope = checkpoint.get('checkpoint_training_scope')
    if scope != expected:
        raise ValueError(f'Checkpoint scope {scope!r} requires its explicit trainer/migration; admitted {expected!r}')

