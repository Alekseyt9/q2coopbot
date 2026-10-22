"""Real displacement coverage in closed rules demonstrations; no fitting."""
import argparse
import collections
import gzip
import json
import math
import pathlib
from process_combat_architecture_pool import read,save,sha


def key(seed,identity):
    return (seed,)+tuple(identity[k] for k in ('map','connection','spawncount','actor','life','frame'))


def main():
    ap=argparse.ArgumentParser()
    for name in ('corpus','features','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    args=ap.parse_args();assert not args.out.exists();pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path).resolve();digest=sha(path);assert expected is None or expected==digest
        pins[str(path)]=digest;return path
    closed=read(pin(args.corpus/'complete.json'));meta=read(pin(args.features/'complete.json'))
    assert closed['state']=='complete' and closed['version']=='registered_rules_teacher_corpus_v1'
    assert meta['state']=='complete' and meta['version']=='registered_joint_teacher_features_v1'
    assert pathlib.Path(meta['corpus']).resolve()==args.corpus.resolve() and not meta['training_performed']
    for binding in closed['inputs']+meta['inputs']:pin(binding['path'],binding['sha256'])
    lookup={}
    for split,info in meta['splits'].items():
        with gzip.open(pin(info['path'],info['sha256']),'rt',encoding='utf-8') as stream:
            for row in map(json.loads,stream):
                k=key(row['seed'],row['identity']);assert k not in lookup;lookup[k]=row
    cases=[]
    for case in closed['cases']:
        report=read(pathlib.Path(case['root'])/'report.json');worker=pathlib.Path(report['results'][0]['root'])
        with (worker/'bot.jsonl').open(encoding='utf-8-sig') as stream:raw=[r['combat_policy'] for r in map(json.loads,stream)]
        counts=collections.Counter();reasons=collections.Counter()
        for before,after in zip(raw,raw[1:]):
            o,n=before['observation'],after['observation'];k=key(case['seed'],o['identity']);feature=lookup.get(k)
            if feature is None:continue
            assert before['applied']['identity']==o['identity']
            if not feature['native_matched'] or not feature['native_exclusive'] or not o['on_ground']:continue
            expected=dict(o['identity'],frame=o['identity']['frame']+1)
            if n['identity']!=expected or n['health']<=0:continue
            counts['eligible_grounded_frames']+=1
            a=before['applied'];requested=math.hypot(a['forward'],a['side'])>.1
            displacement=math.hypot(n['position'][0]-o['position'][0],n['position'][1]-o['position'][1])
            moving=displacement>2.
            counts['requested_horizontal_frames']+=requested
            counts['requested_and_displaced_gt2']+=requested and moving
            counts['requested_and_next_speed_lt10']+=requested and math.hypot(*n['velocity'][:2])<10
            counts['no_horizontal_request_frames']+=not requested
            counts['body_displaced_without_request_gt2']+=not requested and moving
            counts['vertical_'+a['vertical']]+=1
            if requested:reasons[before.get('move_limit_reason') or 'no_move_limit']+=1
        cases.append(dict(seed=case['seed'],split=case['split'],map=raw[0]['observation']['identity']['map'],counts=dict(counts),motion_limit_reasons=dict(reasons)))
    totals={s:dict(sum((collections.Counter(c['counts']) for c in cases if c['split']==s),collections.Counter())) for s in meta['splits']}
    for path,digest in pins.items():assert sha(path)==digest
    save(args.out,dict(state='complete',version='registered_teacher_motion_coverage_v1',cases=cases,totals=totals,
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),training_performed=False,neural_execution=False,
        scope='Grounded adjacent alive first-life rules commands with sealed Go native matched/exclusive proof. Requested movement is distinguished from observed next-frame horizontal displacement>2 and speed. Coverage evidence, not optimal motion labels, jump competence or learned student quality.'))
    print(json.dumps(dict(cases=len(cases),totals=totals)))


if __name__=='__main__':main()
