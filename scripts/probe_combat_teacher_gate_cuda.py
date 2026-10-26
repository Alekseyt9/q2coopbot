"""Frozen-representation paired-preference CV on CUDA, never a policy export."""
import argparse
import copy
from pathlib import Path

from combat_native_teacher_registry import load_index
from combat_teacher_folds import map_folds, world_weights
from combat_wide_aim import validate
from ppo_combat import torch, training_devices
from process_combat_architecture_pool import read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = a.out.resolve()
    assert not out.exists()
    records = load_index(repo, a.index)
    labelled = [r for r in records if r['target']['preference'] is not None]
    source = read(a.source)
    validate(source)
    assert all(r['source_model_sha256'] == sha(a.source) for r in records)
    pins = {str(p.resolve()): sha(p) for p in (a.index, a.source, Path(__file__), repo/'scripts/combat_teacher_folds.py')}
    for corpus in read(a.index)['corpora']:
        path = (repo/corpus['manifest']).resolve()
        pins[str(path)] = sha(path)
        for entry in read(path)['entries']:
            path = Path(entry['path']).resolve()
            pins[str(path)] = sha(path)
    out.mkdir(parents=True)
    protocol = dict(version='combat_teacher_gate_learnability_v1', device='cuda',
        labels=len(labelled), total_records=len(records), maps=len({r['map'] for r in labelled}),
        negative=sum(r['target']['preference'] == 0 for r in labelled),
        positive=sum(r['target']['preference'] == 1 for r in labelled),
        minimum_labels=40, minimum_per_class=10, minimum_maps=16,
        folds=4, steps=300, lr=.03, weight_delta_ridge=.1, std_floor=.05,
        fit_parameters=dict(contextual=17, bias_only=1), inputs=pins,
        target='Original paired gamma.99 observed-return preference, no relabelling.',
        split='Map-grouped internal cross-validation of TRAIN teachers, not globally unseen policy maps or combat-quality evaluation. All roots of a map stay together.',
        frozen='Source policy and187→32→16 representation; only final-layer deltas fitted. No actor weights exported.',
        normalizer='World-balanced TRAIN-only hidden mean/std, floor.05. Folds never share maps.',
        advance_gate='At least5% pooled OOF world-weighted log-loss improvement over bias-only and balanced accuracy>=.55.',
        diagnostic_only=True, actor_compatible=False, promotion=False)
    save(out/'protocol.json', protocol)
    if len(labelled) < 40 or min(protocol['negative'], protocol['positive']) < 10 or protocol['maps'] < 16:
        save(out/'report.json', dict(state='insufficient_teacher_coverage', training_performed=False,
            labels=len(labelled), negative=protocol['negative'], positive=protocol['positive'],
            maps=protocol['maps'], advance_to_native=False, promotion=False))
        print(out/'report.json')
        return
    training_devices()
    torch.set_num_threads(2)
    torch.manual_seed(41)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    functional = torch.nn.functional
    dtype, device = torch.float64, 'cuda'
    context = torch.tensor([r['input']['pre_gate_context187'] for r in labelled], device=device, dtype=dtype)
    y = torch.tensor([r['target']['preference'] for r in labelled], device=device, dtype=dtype)
    with torch.no_grad():
        hidden = context
        for layer in source['wide_aim']['layers'][:2]:
            w = torch.tensor(layer['weight'], device=device, dtype=dtype)
            b = torch.tensor(layer['bias'], device=device, dtype=dtype)
            hidden = functional.relu(functional.linear(hidden, w, b))
        layer = source['wide_aim']['layers'][2]
        base = functional.linear(hidden, torch.tensor(layer['weight'], device=device, dtype=dtype),
                                 torch.tensor(layer['bias'], device=device, dtype=dtype))[:,0]
    assert hidden.shape == (len(labelled), 16) and bool(torch.isfinite(hidden).all())
    fold_ids = map_folds(labelled)
    save(out/'splits.json', dict(records=[dict(id=r['id'], map=r['map'], seed=r['seed'], fold=f)
                                        for r,f in zip(labelled, fold_ids)]))
    predictions = {key: torch.empty_like(y) for key in ('source','bias_only','contextual')}
    predictions['source'].copy_(base.clamp(-20,20))
    fold_reports = []

    def fit(offset, design, target, weights, steps, checkpoint=None):
        theta = torch.nn.Parameter(torch.zeros(design.shape[1], device=device, dtype=dtype))
        optimizer = torch.optim.Adam([theta], lr=.03, foreach=False, capturable=True)
        previous = 0
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            assert state['version'] == 'combat_teacher_gate_diagnostic_checkpoint_v1'
            assert state['actor_compatible'] is False and state['protocol_sha256'] == sha(out/'protocol.json')
            with torch.no_grad():
                theta.copy_(state['theta'])
            optimizer.load_state_dict(state['optimizer'])
            previous = state['steps']
        weights = weights/weights.sum()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            logits = (offset+design@theta).clamp(-20,20)
            loss = (weights*functional.binary_cross_entropy_with_logits(logits,target,reduction='none')).sum()
            loss = loss+.05*theta[:-1].square().sum()
            assert bool(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
        snapshot = dict(version='combat_teacher_gate_diagnostic_checkpoint_v1', actor_compatible=False,
            protocol_sha256=sha(out/'protocol.json'), steps=previous+steps,
            theta=theta.detach().clone(), optimizer=copy.deepcopy(optimizer.state_dict()))
        return theta.detach().clone(), snapshot

    def exact(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b,torch.Tensor) and a.dtype == b.dtype and torch.equal(a,b)
        if isinstance(a,dict):
            return isinstance(b,dict) and a.keys()==b.keys() and all(exact(v,b[k]) for k,v in a.items())
        if isinstance(a,(list,tuple)):
            return type(a) is type(b) and len(a)==len(b) and all(exact(v,w) for v,w in zip(a,b))
        return a == b

    for fold in range(4):
        train = [i for i,f in enumerate(fold_ids) if f != fold]
        test = [i for i,f in enumerate(fold_ids) if f == fold]
        assert train and test and {int(y[i]) for i in train} == {0,1}
        weights = torch.tensor(world_weights([labelled[i] for i in train]),device=device,dtype=dtype)
        normalized = weights/weights.sum()
        mean = (hidden[train]*normalized[:,None]).sum(0)
        std = ((hidden[train]-mean).square()*normalized[:,None]).sum(0).sqrt().clamp_min(.05)
        z = (hidden-mean)/std
        fold_report = dict(fold=fold,train_rows=len(train),heldout_rows=len(test),
                           heldout_maps=sorted({labelled[i]['map'] for i in test}), fits={})
        for kind in ('bias_only','contextual'):
            design = torch.ones((len(labelled),1),device=device,dtype=dtype)
            if kind == 'contextual':
                design = torch.cat((z,design),-1)
            theta, snapshot = fit(base[train],design[train],y[train],weights,300)
            if fold == 0:
                _, partial = fit(base[train],design[train],y[train],weights,150)
                path = out/(kind+'-resume150.pt')
                torch.save(partial,path)
                resumed,resumed_snapshot = fit(base[train],design[train],y[train],weights,150,path)
                assert torch.equal(theta,resumed) and exact(snapshot,resumed_snapshot),'GPU resume differs'
            path = out/f'fold-{fold}-{kind}.pt'
            torch.save(dict(**snapshot, normalizer_mean=mean,normalizer_std=std),path)
            predictions[kind][test] = (base[test]+design[test]@theta).clamp(-20,20)
            fold_report['fits'][kind] = dict(checkpoint=str(path),checkpoint_sha256=sha(path),
                                             train_parameters=len(theta), resume_verified=fold==0)
        fold_reports.append(fold_report)
        save(out/'progress.json',dict(stage='cross_validation',completed_folds=fold+1))
    weights = torch.tensor(world_weights(labelled),device=device,dtype=dtype)
    def metrics(logits):
        prediction = logits>=0
        loss = (weights*functional.binary_cross_entropy_with_logits(logits,y,reduction='none')).sum()/weights.sum()
        recalls = [float((weights[y==cls]*(prediction[y==cls]==bool(cls))).sum()/weights[y==cls].sum()) for cls in (0,1)]
        return dict(log_loss=float(loss),balanced_accuracy=sum(recalls)/2,
                    choice0_recall=recalls[0],choice1_recall=recalls[1])
    scores = {key:metrics(value) for key,value in predictions.items()}
    ratio = scores['contextual']['log_loss']/scores['bias_only']['log_loss']
    advance = ratio<=.95 and scores['contextual']['balanced_accuracy']>=.55
    for path,digest in pins.items():
        assert sha(path)==digest,path
    save(out/'report.json',dict(state='complete',device=device,records=len(records),labels=len(labelled),
        maps=protocol['maps'],negative=protocol['negative'],positive=protocol['positive'],scores=scores,
        contextual_to_bias_loss_ratio=ratio,advance_to_native=advance,
        first_fold_resume_exact=True,folds=fold_reports,
        predictions=[dict(id=r['id'],map=r['map'],seed=r['seed'],frame=r['frame'],label=r['target']['preference'],fold=f,
                          probabilities={key:float(value[i].sigmoid()) for key,value in predictions.items()})
                     for i,(r,f) in enumerate(zip(labelled,fold_ids))],
        scope=protocol['split'],diagnostic_only=True,actor_compatible=False,
        training_performed=True,policy_training_performed=False,promotion=False))
    save(out/'progress.json',dict(stage='complete'))
    print(out/'report.json')


if __name__ == '__main__':
    main()
