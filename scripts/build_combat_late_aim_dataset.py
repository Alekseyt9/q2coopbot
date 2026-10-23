"""Append accepted later contexts while preserving every parent context/split."""
import argparse
import collections
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('parent','selection','proof','protocol','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    ps=read(pin(a.parent/'complete.json'));parent=read(pin(a.parent/'report.json',ps['report_sha256']))
    prior=list(map(json.loads,pin(a.parent/'contexts.jsonl',ps['labels_sha256']).open(encoding='utf-8-sig')))
    ss=read(pin(a.selection/'complete.json'));selection=read(pin(a.selection/'report.json',ss['report_sha256']))
    seal=read(pin(a.proof/'complete.json'));proof=read(pin(a.proof/'report.json',seal['report_sha256']))
    new=list(map(json.loads,pin(a.proof/'labels.jsonl',seal['labels_sha256']).open(encoding='utf-8-sig')));pin(a.protocol)
    assert parent['contexts']==17 and selection['parent_dataset']==str(a.parent.resolve())
    assert parent['source_weights_sha256']==selection['source_weights_sha256']
    source_path=str((Path(selection['source'])/'weights.json').resolve())
    assert proof['inputs'][source_path]==parent['source_weights_sha256']
    expected={(c['seed'],c['frame']):c for c in selection['contexts']};assert len(expected)==len(selection['contexts'])
    accepted=[];excluded=[]
    for r in new:
        c=expected[(r['seed'],r['frame'])];assert r['version']=='combat_native_aim_pair_labels_v1'
        assert r['split']==c['split']
        expected_purpose='train_acquisition' if c['split']=='train' else 'development_labels_only'
        assert r['purpose']==expected_purpose
        if len(r['alternatives'])<3:
            excluded.append(dict(seed=r['seed'],frame=r['frame'],reason='fewer than3 accepted native alternatives'));continue
        assert sum(v['category']==0 for v in r['alternatives'])==1 and len(r['features'])==1147
        r.update(cohort='late-v1',label_proof=str(a.proof.resolve()),label_proof_sha256=seal['report_sha256']);accepted.append(r)
    available={(r['seed'],r['frame']) for r in new}
    for key,c in expected.items():
        if key not in available:excluded.append(dict(seed=key[0],frame=key[1],reason='baseline/prefix not accepted'))
    rows=prior+accepted;assert len({(r['seed'],r['frame']) for r in rows})==len(rows)
    assert max(collections.Counter(r['seed'] for r in rows).values())<=2
    heldout=set(selection['heldout_seeds']);assert set(parent['heldout_seeds'])<=heldout
    assert all((r['seed'] in heldout)==(r['split']=='development') for r in rows)
    assert rows[:len(prior)]==prior
    statistics={}
    for split in ('train','development'):
        group=[r for r in rows if r['split']==split];fresh=[r for r in accepted if r['split']==split]
        positive=lambda r:any(v['health_damage']>r['baseline_health_damage'] for v in r['alternatives'])
        statistics[split]=dict(contexts=len(group),seeds=sorted({r['seed'] for r in group}),new_contexts=len(fresh),new_seeds=sorted({r['seed'] for r in fresh}),
            improving_contexts=sum(positive(r) for r in group),new_improving_contexts=sum(positive(r) for r in fresh),
            improving_seeds=sorted({r['seed'] for r in group if positive(r)}),
            alternatives=sum(len(r['alternatives']) for r in group),baseline_hit_contexts=sum(r['baseline_health_damage']>0 for r in group))
    reasons=[]
    if any(statistics[s]['new_contexts']<6 or len(statistics[s]['new_seeds'])<4 for s in statistics):reasons.append('Insufficient new context/seed coverage in a split')
    if len(statistics['train']['improving_seeds'])<2:reasons.append('Insufficient independent TRAIN improvements')
    for p,h in pins.items():assert sha(Path(p))==h,p
    a.out.mkdir(parents=True);dataset=a.out/'contexts.jsonl';dataset.write_text(''.join(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n' for r in rows),encoding='utf-8')
    report=dict(state='complete',version='combat_native_aim_pair_dataset_v3_late',contexts=len(rows),parent_dataset=str(a.parent.resolve()),
        parent_contexts=17,new_contexts=len(accepted),excluded=excluded,source_weights_sha256=parent['source_weights_sha256'],
        statistics=statistics,heldout_seeds=sorted(heldout),labels_sha256=sha(dataset),fit_admission_passed=not reasons,fit_admission_reasons=reasons,
        parent_contexts_and_splits_unchanged=True,inputs=pins,script_sha256=sha(__file__),training_performed=False,validation_used=False,promotion=False,
        scope='Registry TRAIN with preserved parent and preassigned fresh whole-seed development. Acquisition labels only, no ordinary PPO rollout or combat quality claim.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json'),labels_sha256=sha(dataset)))
    print(json.dumps({k:report[k] for k in ('contexts','new_contexts','statistics','fit_admission_passed','excluded')},indent=2))


if __name__=='__main__':main()
