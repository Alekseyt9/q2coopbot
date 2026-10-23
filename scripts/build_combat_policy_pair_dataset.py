"""Join accepted one-frame policy labels and observed returns without split changes."""
import argparse
import copy
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('selection','protocol','out'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--proof',type=Path,action='append',required=True)
    ap.add_argument('--returns',type=Path,action='append',required=True)
    a=ap.parse_args();assert not a.out.exists() and len(a.proof)==len(a.returns);pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    ss=read(pin(a.selection/'complete.json'));selection=read(pin(a.selection/'report.json',ss['report_sha256']))
    assert selection['version']=='combat_late_aim_selection_v2_policy';pin(a.protocol)
    source=str((Path(selection['source'])/'weights.json').resolve());pin(source,selection['source_weights_sha256'])
    expected={(c['seed'],c['frame']):c for c in selection['contexts']};seen=set();rows=[];return_rows=[];excluded=[];excluded_actions=[]
    for proof_dir,return_dir in zip(a.proof,a.returns):
        ps=read(pin(proof_dir/'complete.json'));proof=read(pin(proof_dir/'report.json',ps['report_sha256']))
        assert proof['one_frame_probe'] and proof['inputs'][source]==selection['source_weights_sha256']
        labels=list(map(json.loads,pin(proof_dir/'labels.jsonl',ps['labels_sha256']).open(encoding='utf-8-sig')))
        rs=read(pin(return_dir/'complete.json'));rr=read(pin(return_dir/'report.json',rs['report_sha256']))
        assert rr['version']=='combat_one_frame_observed_returns_v1' and rr['inputs'][str((proof_dir/'report.json').resolve())]==ps['report_sha256']
        for p,h in rr['inputs'].items():pin(p,h)
        labelmap={(r['seed'],r['frame']):r for r in labels};returnmap={(r['seed'],r['frame']):r for r in rr['contexts']}
        assert len(labelmap)==len(labels) and len(returnmap)==len(rr['contexts'])
        for c in proof['contexts']:
            key=c['seed'],c['frame'];assert key in expected and key not in seen;seen.add(key);selected=expected[key]
            if not c['full_baseline_trajectory_reproduced'] or key not in labelmap or key not in returnmap:
                excluded.append(dict(seed=key[0],frame=key[1],reason='No accepted full baseline/labels/returns'));continue
            r=copy.deepcopy(labelmap[key]);ret=copy.deepcopy(returnmap[key]);assert r['split']==ret['split']==selected['split']
            assert r['purpose']==('train_acquisition' if r['split']=='train' else 'development_labels_only')
            assert len(r['features'])==1147 and r['geometry']==selected['geometry']
            bycategory={v['category']:v for v in ret['samples']};actions=[];matched=[]
            for v in r['alternatives']:
                if v['category'] not in bycategory:
                    excluded_actions.append(dict(seed=key[0],frame=key[1],category=v['category'],reason='No accepted reward prefix'));continue
                rv=bycategory[v['category']];assert rv['immediate_hp']==v['health_damage']
                actions.append(dict(v,full_delta=rv['full_delta'],sparse_delta=rv['sparse_delta']));matched.append(rv)
            if len(actions)<3 or sum(v['category']==0 for v in actions)!=1:
                excluded.append(dict(seed=key[0],frame=key[1],reason='Fewer than3 common actions or no baseline'));continue
            assert len({v['category'] for v in actions})==len(actions)
            r.update(alternatives=actions,cohort='policy-v2',source_policy_sha256=selection['source_weights_sha256'],label_proof=str(proof_dir.resolve()))
            ret['samples']=matched;rows.append(r);return_rows.append(ret)
    assert seen==set(expected),('Acquisition proofs do not cover selection',set(expected)-seen)
    assert len({r['seed'] for r in rows})==len(rows)
    trainseeds={r['seed'] for r in rows if r['split']=='train'};devseeds={r['seed'] for r in rows if r['split']=='development'}
    assert not trainseeds&devseeds and all((r['seed'] in selection['heldout_seeds'])==(r['split']=='development') for r in rows)
    statistics={}
    for split in ('train','development'):
        group=[r for r in rows if r['split']==split];mp=lambda r:r['geometry'].split('-site-')[0]
        immediate=[r for r in group if any(v['health_damage']>r['baseline_health_damage'] for v in r['alternatives'])]
        sparse=[r for r in group if any(v['sparse_delta']>=1 for v in r['alternatives'])]
        statistics[split]=dict(contexts=len(group),seeds=sorted(r['seed'] for r in group),maps=sorted({mp(r) for r in group}),
            alternatives=sum(len(r['alternatives']) for r in group),improving_contexts=len(immediate),improving_maps=sorted({mp(r) for r in immediate}),
            sparse_improving_contexts=len(sparse),sparse_improving_maps=sorted({mp(r) for r in sparse}))
    base_reasons=[]
    if statistics['train']['contexts']<16 or statistics['development']['contexts']<6:base_reasons.append('Insufficient independent contexts')
    if any(len(s['maps'])<3 for s in statistics.values()):base_reasons.append('Insufficient map coverage in a split')
    reasons={n:list(base_reasons) for n in ('immediate','sparse_return')}
    t=statistics['train']
    if t['improving_contexts']<4 or len(t['improving_maps'])<2:reasons['immediate'].append('Insufficient independent immediate improvements')
    if t['sparse_improving_contexts']<4 or len(t['sparse_improving_maps'])<2:reasons['sparse_return'].append('Insufficient independent return improvements')
    for p,h in pins.items():assert sha(p)==h,p
    a.out.mkdir(parents=True);labels=a.out/'contexts.jsonl';labels.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows),encoding='utf-8')
    rd=a.out/'returns';rd.mkdir();save(rd/'report.json',dict(state='complete',version='combat_one_frame_observed_returns_v1',contexts=return_rows,
        excluded=[],upstream_excluded_contexts=excluded,upstream_excluded_actions=excluded_actions,inputs=pins,training_performed=False,promotion=False,
        scope='Accepted common policy-continuation actions only. Upstream exclusions preserved; no expected Q or validation labels.'))
    save(rd/'complete.json',dict(state='complete',report_sha256=sha(rd/'report.json')))
    admissions={n:not v for n,v in reasons.items()}
    save(a.out/'report.json',dict(state='complete',version='combat_native_aim_pair_dataset_v4_policy',contexts=len(rows),statistics=statistics,
        source_weights_sha256=selection['source_weights_sha256'],heldout_seeds=selection['heldout_seeds'],labels_sha256=sha(labels),
        fit_admission_passed=admissions['immediate'],fit_admissions=admissions,fit_admission_reasons=reasons,matched_control_required=True,excluded=excluded,excluded_actions=excluded_actions,
        returns_report_sha256=sha(rd/'report.json'),inputs=pins,script_sha256=sha(__file__),training_performed=False,validation_used=False,promotion=False))
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json'),labels_sha256=sha(labels),returns_report_sha256=sha(rd/'report.json')))
    print(json.dumps(dict(contexts=len(rows),statistics=statistics,fit_admissions=admissions,reasons=reasons,excluded=excluded,excluded_actions=excluded_actions),indent=2))


if __name__=='__main__':main()
