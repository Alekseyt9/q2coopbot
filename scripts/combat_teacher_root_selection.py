"""Outcome-blind temporal coverage for one paired root per source world per batch."""


def choose_root_frame(frames, used_frames=(), minimum_gap=30):
    """Use the earliest eligible event, or advance past all registered roots.

    Inputs contain only frame identities, never utility, labels, or outcomes.
    Repeated contexts from one world remain correlated training observations.
    """
    assert type(minimum_gap) is int and minimum_gap > 0
    frames, used_frames = list(frames), list(used_frames)
    assert all(type(f) is int and f >= 110 for f in frames)
    assert all(type(f) is int and f > 0 for f in used_frames)
    threshold = max(used_frames) + minimum_gap if used_frames else 110
    return next((f for f in sorted(set(frames)) if f >= threshold), None)
