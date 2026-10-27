"""Metadata regression checks on real paired captures. No neural computation."""
import argparse
import copy
import gzip
import json
import tempfile
from pathlib import Path
from combat_causal_fire_pairs import load, validate, preferred_attack
from process_combat_architecture_pool import sha, save


def check(path, out):
    assert not out.exists()
    data=load(path)
    checks={'real_provenance':True}
    def rejects(name,mutation):
        bad=copy.deepcopy(data)
        mutation(bad)
        try:validate(bad)
        except (AssertionError,KeyError,TypeError,ValueError):checks[name]=True
        else:raise AssertionError('Accepted invalid data: '+name)
    rejects('legacy_format',lambda d:d.update(version='combat_causal_fire_pairs_v1'))
    rejects('ordinary_ppo',lambda d:d.update(ordinary_ppo_eligible=True))
    rejects('duplicate_world',lambda d:d['cases'].append(copy.deepcopy(d['cases'][0])))
    rejects('invalid_split',lambda d:d['cases'][0].update(split='validation'))
    rejects('future_feature',lambda d:d['cases'][0]['input'].update(next_features=[]))
    rejects('future_action_input',lambda d:d['cases'][0]['input']['sample'].update(reward=1))
    rejects('future_frame',lambda d:d['cases'][0]['input']['sequence_prefix'][-1].update(frame=d['cases'][0]['input']['frame']+1))
    rejects('future_index',lambda d:d['cases'][0]['input']['sequence_prefix'][-1].update(index=d['cases'][0]['input']['index']+1))
    rejects('missing_reset',lambda d:d['cases'][0]['input']['sequence_prefix'][0]['memory'].update(reset=False))
    rejects('nonfinite_memory',lambda d:d['cases'][0]['input']['sequence_prefix'][0]['memory']['actor'].append(float('nan')))
    rejects('nonfinite_feature',lambda d:d['cases'][0]['input']['features'].__setitem__(0,float('nan')))
    rejects('nonfinite_return',lambda d:d['cases'][0]['target'].update(source_return=float('inf')))
    rejects('nonfinite_margin',lambda d:d['label_config'].update(margin=float('inf')))
    rejects('flipped_label',lambda d:d['cases'][0]['target'].update(preferred_attack=1-d['cases'][0]['target']['preferred_attack']))
    rejects('boolean_label',lambda d:d['cases'][0]['target'].update(preferred_attack=bool(d['cases'][0]['target']['preferred_attack'])))
    rejects('invalid_target',lambda d:d['cases'][0]['input']['sample'].update(target=9))
    rejects('ownership_bypass',lambda d:next(c for c in d['cases'] if not c['eligible_for_preference']).update(eligible_for_preference=True))
    rejects('invalid_ownership',lambda d:d['cases'][0]['ownership'].update(source=[-1]))
    assert preferred_attack(.1,.1) is None and preferred_attack(-.1,.1) is None
    assert preferred_attack(.11,.1)==0 and preferred_attack(-.11,.1)==1
    checks['margin_boundary']=True
    summary={s:{'cases':sum(c['split']==s for c in data['cases']),
        'eligible':sum(c['split']==s and c['eligible_for_preference'] for c in data['cases'])} for s in ('train','diagnostic')}
    assert summary=={'train':{'cases':6,'eligible':6},'diagnostic':{'cases':3,'eligible':2}}
    checks['all_worlds_retained']=True
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        source=Path(temp)/'source.txt';source.write_text('before',encoding='utf-8')
        bad=copy.deepcopy(data);bad['inputs']={str(source):sha(source)}
        candidate=Path(temp)/'pairs.json.gz'
        with gzip.open(candidate,'wt',encoding='utf-8') as f:json.dump(bad,f)
        load(candidate)
        source.write_text('after',encoding='utf-8')
        try:load(candidate)
        except AssertionError:checks['changed_source_rejected']=True
        else:raise AssertionError('Changed source accepted')
    save(out,dict(state='passed',checks=checks,summary=summary,dataset=str(path.resolve()),
        dataset_sha256=sha(path),no_neural_computation=True))
    print(json.dumps({'state':'passed','checks':len(checks),'summary':summary}))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();check(a.dataset,a.out)
