"""Combine sealed native pair labels and freeze seed-group development splits."""
import argparse
import collections
import json
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--proof',action='append',type=Path,required=True)
    for n in ('protocol','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={};rows=[];sources=[]
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    pin(a.protocol)
    for root in a.proof:
        seal=read(pin(root/'complete.json'));report=read(pin(root/'report.json',seal['report_sha256']))
        assert report['state']=='complete' and not report['training_performed'] and not report['validation_used']
        labels=pin(root/'labels.jsonl',seal['labels_sha256']);items=list(map(json.loads,labels.open(encoding='utf-8-sig')))
        assert len(items)==report['label_contexts']
        source_paths=[p for p in report['inputs'] if p.endswith('weights.json') and 'baseline-model' in p]
        assert len(source_paths)==1;sources.append(report['inputs'][source_paths[0]])
        for item in items:
            assert item['version']=='combat_native_aim_pair_labels_v1' and len(item['features'])==1147
            assert len(item['alternatives'])>=3 and sum(v['category']==0 for v in item['alternatives'])==1
            baseline=next(v for v in item['alternatives'] if v['category']==0)
            assert baseline['health_damage']==item['baseline_health_damage']
            item['label_proof']=str(root.resolve());item['label_proof_sha256']=seal['report_sha256'];rows.append(item)
    assert len(set(sources))==1,'Different source weights require a separately justified data contract'
    assert len(rows)==16 and len({(r['seed'],r['frame']) for r in rows})==16
    assert len({r['context_id'] for r in rows})==16
    geometries=sorted({r['geometry'] for r in rows});assert len(geometries)==4
    assert all(sum(r['geometry']==g for r in rows)==4 for g in geometries)
    seeds=collections.Counter(r['seed'] for r in rows);assert max(seeds.values())<=2
    heldout={max(r['seed'] for r in rows if r['geometry']==g) for g in geometries};assert len(heldout)==4
    for r in rows:r['split']='development' if r['seed'] in heldout else 'train'
    statistics={}
    for split in ('train','development'):
        group=[r for r in rows if r['split']==split]
        statistics[split]=dict(contexts=len(group),seeds=sorted({r['seed'] for r in group}),
            improving_contexts=sum(any(v['health_damage']>r['baseline_health_damage'] for v in r['alternatives']) for r in group),
            regressing_contexts=sum(any(v['health_damage']<r['baseline_health_damage'] for v in r['alternatives']) for r in group),
            baseline_hit_contexts=sum(r['baseline_health_damage']>0 for r in group),
            alternatives=sum(len(r['alternatives']) for r in group))
    reasons=[]
    if statistics['train']['improving_contexts']<2:reasons.append('Fewer than two TRAIN contexts have an observed improvement over baseline')
    if statistics['development']['improving_contexts']<1:reasons.append('No development context has an observed improvement over baseline')
    for p,h in pins.items():assert sha(Path(p))==h,p
    a.out.mkdir(parents=True);dataset=a.out/'contexts.jsonl'
    dataset.write_text(''.join(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n' for r in rows),encoding='utf-8')
    report=dict(state='complete',version='combat_native_aim_pair_dataset_v1',contexts=16,geometries=geometries,
        source_weights_sha256=sources[0],statistics=statistics,heldout_seeds=sorted(heldout),
        split_rule='Maximum selected seed per geometry, whole seed assigned to development. Fixed before alternative labels.',
        protocol_sha256=sha(a.protocol),fit_admission_passed=not reasons,fit_admission_reasons=reasons,
        labels_sha256=sha(dataset),inputs=pins,script_sha256=sha(__file__),training_performed=False,
        validation_used=False,promotion=False,quality_confirmation=False,
        scope='Reusable targeted TRAIN counterfactual dataset, fixed seed groups. Development data are not training and are not independent combat validation. No on-policy PPO use.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json'),labels_sha256=sha(dataset)))
    print(json.dumps({k:report[k] for k in ('contexts','statistics','heldout_seeds','fit_admission_passed','fit_admission_reasons')},indent=2))


if __name__=='__main__':main()
