"""Decision-cadence returns; never repeat an option's categorical loss per frame."""


def training_windows(rows, context, bootstrap_values=None):
    """Explicitly censor at a decision boundary, retaining every case and receipt.

    A nonterminal partial tail cannot use the decision value at a held state.
    Move only the loss horizon back to the start of that last option. The prior
    complete option then bootstraps from the recorded value at that decision.
    All provider history remains available for causal replay.
    """
    bootstrap_values=bootstrap_values or {}
    lookup={(c['seed'],c['frame']):c for c in context}
    kept=[];tails=[];position=0;seen=set()
    while position<len(rows):
        start=position;seed=rows[start]['seed'];seen.add(seed);position+=1
        while position<len(rows) and rows[position]['seed']==seed:
            previous=rows[position-1]
            assert not previous['bootstrap_zero'],'Rows after a terminal first-life transition'
            if rows[position]['frame']!=previous['next_frame']:break
            position+=1
        block=rows[start:position]
        # Only executed decision roots may own a categorical loss. A resumed
        # trace can start inside an option whose root transition was omitted.
        prefix=0
        while prefix<len(block) and not block[prefix]['sample']['movement_option']['decision']:prefix+=1
        if prefix:
            tails.append(dict(seed=seed,frames=[r['frame'] for r in block[:prefix]],
                reason='held_option_prefix_without_admitted_decision'))
            block=block[prefix:]
        if not block:continue
        last=block[-1]
        successor=lookup.get((seed,last['next_frame']))
        restored=(seed,last['next_frame']) in bootstrap_values
        if not last['bootstrap_zero'] and not restored and (successor is None or not successor['movement_option']['decision']):
            boundary=last['sample']['movement_option']['started_frame']
            tail=[r for r in block if r['frame']>=boundary]
            assert 1<=len(tail)<5 and tail[0]['sample']['movement_option']['decision']
            assert not any(r['bootstrap_zero'] for r in tail)
            tails.append(dict(seed=seed,frames=[r['frame'] for r in tail],
                reason='nonterminal_partial_option_loss_horizon',bootstrap_frame=boundary))
            block=block[:-len(tail)]
        kept.extend(block)
    groups=windows(kept,context,bootstrap_values)
    assert len(kept)+sum(len(t['frames']) for t in tails)==len(rows)
    assert {r['seed'] for r in kept}==seen,'Case contains no complete or terminal movement option'
    return kept,groups,tails


def windows(rows, context, bootstrap_values=None):
    bootstrap_values=bootstrap_values or {}
    lookup = {(c['seed'], c['frame']): c for c in context}
    assert len(lookup) == len(context)
    result = []; position = 0
    while position < len(rows):
        start = position; root = rows[start]; option = root['sample']['movement_option']
        assert option['decision'] and option['started_frame'] == root['frame'], 'Loss starts inside a held option'
        while True:
            row = rows[position]; held = row['sample']['movement_option']
            assert row['seed'] == root['seed'] and row['frame'] == root['frame']+position-start
            assert held['started_frame'] == option['started_frame'] and held['choice'] == option['choice']
            assert held['decision'] == (position == start)
            assert row['next_frame'] == row['frame']+1, 'Non-unit native reward transition'
            position += 1
            if row['bootstrap_zero'] or position-start == 5 or position == len(rows): break
            following = rows[position]
            if following['seed'] != root['seed'] or following['frame'] != row['next_frame']: break
            if following['sample']['movement_option']['decision']: break
        last = rows[position-1]; closed = bool(last['bootstrap_zero'])
        successor = lookup.get((root['seed'], last['next_frame']))
        if closed:
            next_value = 0.
        else:
            assert not last['next_reset'], 'Nonterminal reset is not a same-life option bootstrap'
            if successor is None and (root['seed'],last['next_frame']) in bootstrap_values:
                assert position-start==5, 'Reconstructed value is only valid at a decision boundary'
                next_value=bootstrap_values[(root['seed'],last['next_frame'])]
            else:
                assert successor is not None and successor['movement_option']['decision'], \
                    'Nonterminal cut inside held option needs an explicit conditioned bootstrap'
                next_value = successor['movement_option']['value']
        result.append(dict(row=start, count=position-start, seed=root['seed'], frame=root['frame'],
            next_frame=last['next_frame'], closed=closed, old_value=option['value'], next_value=next_value))
    for i, window in enumerate(result):
        following = result[i+1] if i+1 < len(result) else None
        window['continues'] = bool(not window['closed'] and following and
            following['seed']==window['seed'] and following['frame']==window['next_frame'])
        if window['continues']: assert window['next_value'] == following['old_value']
    assert sum(w['count'] for w in result) == len(rows)
    return result


def targets(rows, groups, gamma, lam, torch):
    """All discounted reward/value arithmetic takes place on CUDA."""
    assert 0 < gamma <= 1 and 0 <= lam <= 1
    reward = torch.tensor([r['reward'] for r in rows], device='cuda', dtype=torch.float64)
    gathered = torch.tensor([[w['row']+min(j,w['count']-1) for j in range(5)] for w in groups],
        device='cuda', dtype=torch.long)
    duration = torch.tensor([w['count'] for w in groups], device='cuda', dtype=torch.long)
    offsets = torch.arange(5, device='cuda')
    mask = offsets[None,:] < duration[:,None]
    base = torch.tensor(gamma,device='cuda',dtype=torch.float64)
    discounted = (reward[gathered]*mask*(base**offsets)[None,:]).sum(-1)
    old = torch.tensor([w['old_value'] for w in groups],device='cuda',dtype=torch.float64)
    nxt = torch.tensor([w['next_value'] for w in groups],device='cuda',dtype=torch.float64)
    discount = base**duration
    delta = discounted+discount*nxt-old
    advantage = torch.empty_like(delta); carry = torch.zeros((),device='cuda',dtype=torch.float64)
    for i in range(len(groups)-1,-1,-1):
        carry = delta[i]+discount[i]*lam*carry*float(groups[i]['continues'])
        advantage[i] = carry
    assert bool(torch.isfinite(advantage).all())
    return advantage.float(), (advantage+old).float(), discounted
