"""Explicit attention PPO critic timing experiment; preserves recorded samples."""
def reestimate(rows, prepared, value, gamma, lam, advantages, torch):
    """Evaluate current/next baselines on their true full causal histories."""
    x, indices, inverse, segments = prepared
    locations = {(c['seed'], c['frame']): (b, t)
                 for b, segment in enumerate(segments) for t, c in enumerate(segment)}
    assert len(locations) == sum(map(len, segments))
    with torch.no_grad():
        output, _ = value(x)
        current = output[indices[:, 0], indices[:, 1]][inverse, 0].detach().cpu().tolist()
        successors = []
        for row in rows:
            if row['terminal'] or row.get('bootstrap_zero'):
                successors.append(0.)
                continue
            b, t = locations[(row['seed'], row['next_frame'])]
            successor = segments[b][t]
            assert row['next_features'] == successor['features']
            assert bool(row['next_reset']) == bool(successor['memory']['reset'])
            successors.append(float(output[b, t, 0]))
    # No mutation of behavior values, action probabilities, histories or samples.
    relabeled = [{**row, 'sample': {**row['sample'], 'value': baseline}, 'next_value': successor}
                 for row, baseline, successor in zip(rows, current, successors)]
    adv, _ = advantages(relabeled, gamma, lam)
    return adv, current, successors


def fit(value, optimizer, compute, targets, steps, clip, torch, nn):
    with torch.no_grad():
        before = float((compute(value)[0][:, 0] - targets).square().mean())
    for _ in range(steps):
        prediction, _ = compute(value)
        loss = (prediction[:, 0] - targets).square().mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(value.parameters(), clip); optimizer.step()
    with torch.no_grad():
        after = float((compute(value)[0][:, 0] - targets).square().mean())
    return dict(steps=steps, mse_before=before, mse_after=after)
