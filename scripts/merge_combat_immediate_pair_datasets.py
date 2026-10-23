"""Merge compatible immediate native labels; never merge continuation returns."""
import argparse
import copy
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('old-dataset','old-source','new-dataset','source','protocol','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    sources=[]
    for directory in (a.old_source,a.source):
        seal=read(pin(directory/'complete.json'));sources.append(read(pin(directory/'weights.json',seal['weights_sha256'])))
    left,right=copy.deepcopy(sources[0]),copy.deepcopy(sources[1])
    left['aim_candidate_head']['layers']=right['aim_candidate_head']['layers']=None
    assert left==right,'Different frozen body/predictor; native labels are not compatible'
    pin(a.protocol);rows=[];seed_splits={};keys=set();source_shas=[]
    for directory,model_source,cohort in ((a.old_dataset,a.old_source,'legacy-native'),(a.new_dataset,a.source,'policy-v2')):
        seal=read(pin(directory/'complete.json'));meta=read(pin(directory/'report.json',seal['report_sha256']))
        assert meta['source_weights_sha256']==sha(model_source/'weights.json');source_shas.append(meta['source_weights_sha256'])
        incoming=list(map(json.loads,pin(directory/'contexts.jsonl',seal['labels_sha256']).open(encoding='utf-8-sig')))
        assert len(incoming)==meta['contexts']
        for original in incoming:
            r=copy.deepcopy(original);key=r['seed'],r['frame'];assert key not in keys;keys.add(key)
            assert r['split'] in ('train','development') and seed_splits.get(r['seed'],r['split'])==r['split'];seed_splits[r['seed']]=r['split']
            r['alternatives']=[{k:v[k] for k in ('category','outcome','health_damage','delta_from_baseline')} for v in r['alternatives']]
            assert len({v['category'] for v in r['alternatives']})==len(r['alternatives']) and sum(v['category']==0 for v in r['alternatives'])==1
            r.update(cohort=cohort,origin_dataset=str(directory.resolve()),origin_dataset_sha256=seal['labels_sha256'])
            assert r['features']==original['features'] and r['sample']==original['sample']
            rows.append(r)
    statistics={}
    for split in ('train','development'):
        group=[r for r in rows if r['split']==split]
        positive=[r for r in group if any(v['health_damage']>r['baseline_health_damage'] for v in r['alternatives'])]
        statistics[split]=dict(contexts=len(group),seeds=sorted({r['seed'] for r in group}),maps=sorted({r['geometry'].split('-site-')[0] for r in group}),
             alternatives=sum(len(r['alternatives']) for r in group),improving_seeds=sorted({r['seed'] for r in positive}),
             improving_maps=sorted({r['geometry'].split('-site-')[0] for r in positive}))
    reasons=[];t,d=statistics['train'],statistics['development']
    if t['contexts']<40 or d['contexts']<20 or len(t['seeds'])<30 or len(d['seeds'])<14:reasons.append('Insufficient independent coverage')
    if len(t['maps'])<3 or len(t['improving_seeds'])<4 or len(t['improving_maps'])<2:reasons.append('Insufficient TRAIN map/improvement coverage')
    for p,h in pins.items():assert sha(p)==h,p
    a.out.mkdir(parents=True);path=a.out/'contexts.jsonl';path.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows),encoding='utf-8')
    save(a.out/'report.json',dict(state='complete',version='combat_native_immediate_pair_dataset_v1',contexts=len(rows),statistics=statistics,
         source_weights_sha256=source_shas[1],compatible_source_shas=source_shas,labels_sha256=sha(path),fit_admission_passed=not reasons,
         fit_admission_reasons=reasons,inputs=pins,script_sha256=sha(__file__),training_performed=False,promotion=False,
         scope='Immediate native HP labels only; continuation returns removed. Features/samples/splits preserved; body/predictor identical except candidate head layers. No new games.'))
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json'),labels_sha256=sha(path)))
    print(json.dumps(dict(contexts=len(rows),statistics=statistics,fit_admission_passed=not reasons),indent=2))


if __name__=='__main__':main()
