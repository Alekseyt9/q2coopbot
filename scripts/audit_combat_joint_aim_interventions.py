"""Count sampled corrections only in verified native provider command windows."""
import argparse
import collections
import json
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation','spec','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();digest=sha(p);assert expected is None or digest==expected,str(p)
        pins[str(p)]=digest;return p
    root=a.evaluation.resolve();closed=read(pin(root/'complete.json'))
    assert closed['state']=='complete' and closed['native_cases']==48 and closed['cuda_cases']==40
    spec=read(pin(a.spec,closed['spec_sha256']));pool=read(pin(root/'pool/report.json',closed['pool_sha256']))
    assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==48
    jobs={j['job']:j for j in pool['jobs']};variants={}
    for index,v in enumerate(spec['variants']):
        if v['mode']!='learned':continue
        label=v['label'];execution=read(pin(root/('execution-'+label+'.json')))
        assert execution['state']=='complete' and execution['selected_plan_index']==index and execution['pool_sha256']==closed['pool_sha256']
        cases=[];total=collections.Counter()
        for binding in execution['cases']:
            job=jobs[binding['job']];assert not job['error'] and job['plan_index']==index
            directory=root/('processing-'+label)/f"case-{job['job']:04d}"
            meta=read(pin(directory/'cuda/report.json'))
            verification=read(pin(directory/'cuda/cuda-verification.json'))
            assert meta['numerical_verification']=='cuda_verified_v1' and verification['state']=='passed' and verification['device']=='cuda'
            steps=(directory/'native/replay-0/steps.jsonl').resolve();pin(steps,meta['source_sha256'][str(steps)])
            count=collections.Counter(provider_windows=0,eligible=0,nonzero=0,nonzero_attack=0);events=[];seen=set()
            for s in map(json.loads,steps.open(encoding='utf-8-sig')):
                o=s['observation'];n=s.get('native_step');d=s.get('server_execution') or {};sample=s.get('sample')
                if (not n or not sample or s['owner']!='provider' or o['identity']['life']!=1 or o['health']<=0
                        or o['identity']['frame']<binding['network_release_frame'] or not d.get('matched')
                        or not d.get('window_exclusive') or d.get('recovery_commands')!=0):continue
                assert n['spawncount']==binding['release_generation']
                key=(n['spawncount'],n['actor'],n['sequence']);assert key not in seen;seen.add(key)
                count['provider_windows']+=1;event=sample.get('aim_candidate')
                if not event:continue
                count['eligible']+=1
                if event['candidate']!=0:
                    count['nonzero']+=1;count['nonzero_attack']+=int(sample['attack'])
                    events.append(dict(frame=o['identity']['frame'],candidate=event['candidate'],attack=sample['attack']))
            assert count['provider_windows']>0
            total.update(count);cases.append(dict(job=job['job'],seed=job['seed'],episode=binding.get('episode'),**count,events=events))
        assert len(cases)==8;variants[label]=dict(totals=dict(total),cases=cases)
    gains={}
    for control in ('source','B'):
        q=read(pin(root/f'quality-S-vs-{control}.json'))
        assert q['state']=='complete' and q['before_pool_sha256']==q['after_pool_sha256']==closed['pool_sha256']
        by_seed={c['seed']:c for c in variants['S']['cases']};rows=[]
        for p in q['pairs']:
            if p['metrics']['after']['goal'] and not p['metrics']['before']['goal']:
                case=by_seed[p['seed']]
                rows.append(dict(seed=p['seed'],S_nonzero_choices=case['nonzero'],S_provider_windows=case['provider_windows']))
        gains[control]=rows
    report=dict(state='complete',version='combat_joint_aim_native_interventions_v1',neural_execution=False,new_games=False,
        training_performed=False,variants=variants,S_gained_cases=gains,
        S_all_gains_without_runtime_corrections=all(r['S_nonzero_choices']==0 for rows in gains.values() for r in rows),
        inputs=pins,script_sha256=sha(__file__),causal_improvement_proven=False,promotion=False,
        scope='Actual sampled nonzero categories in alive first-life exclusive unrecovered native provider windows after canonical release. Zero choices preserve source latent exactly. Gains without corrections cannot be attributed to applying a new aim offset at runtime; this does not exclude indirect training effects or establish why the base actor improved.')
    a.out.mkdir(parents=True);save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print({k:v['totals'] for k,v in variants.items()});print(gains)


if __name__=='__main__':main()
