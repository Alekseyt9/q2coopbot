"""Frozen TRAIN shot diagnostics: observed versus shot-time recoil, no NN."""
import argparse
import collections
import copy
import json
from pathlib import Path
from evaluate_registered_combat_models import read,sha
from native_combat_release import release_boundary
from report_combat_machinegun_hits import scan,vector
from report_combat_machinegun_aim import measure,ray


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','noise','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();assert not args.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path);assert expected is None or expected==digest,str(path)
        pins[str(path)]=digest;return path
    noise=read(pin(args.noise/'report.json'))
    assert noise['state']=='complete' and noise['device']=='cuda' and not noise['training_performed']
    for path,digest in noise['inputs'].items():pin(path,digest)
    corpus=args.corpus.resolve();pool=read(pin(corpus/'pool/report.json'))
    plans=[read(pin(p['path'],p['sha256'])) for p in pool['plans']]
    assert all(t['split']=='train' for p in plans for t in p['tasks'])
    data=pin(corpus/'processing-train/merged/rollout.jsonl')
    with data.open(encoding='utf-8-sig') as stream:accepted={(r['seed'],r['index'],r['frame']) for r in map(json.loads,stream)}
    entries=[];statuses=collections.Counter()
    for job in sorted(pool['jobs'],key=lambda j:j['job']):
        task=plans[job['plan_index']]['tasks'][job['task_index']]
        capture=read(pin(Path(job['root'])/'report.json'));result=capture['results'][0]
        assert capture['capture_complete'] and capture['provenance_valid'] and result['capture_valid'] and result['dispatch_valid']
        manifest=read(pin(Path(job['root'])/'manifest.json'));log=pin(Path(result['root'])/'server.log')
        with log.open(encoding='utf-8-sig') as stream:generation,release,_=release_boundary(stream,manifest,job['seed'])
        steps=pin(corpus/'processing-train'/f"case-{job['job']:04d}/native/replay-0/steps.jsonl");windows={}
        with steps.open(encoding='utf-8-sig') as stream:
            for s in map(json.loads,stream):
                o=s['observation'];n=s.get('native_step');d=s.get('server_execution') or {}
                if not n or s['owner']!='provider' or o['identity']['life']!=1 or o['health']<=0 or o['identity']['frame']<release:continue
                if (job['seed'],s['index'],o['identity']['frame']) not in accepted:continue
                if not d.get('matched') or not d.get('window_exclusive') or d.get('recovery_commands')!=0:continue
                assert n['spawncount']==generation
                key=(n['spawncount'],n['begin_frame'],n['actor'],n['sequence']);assert key not in windows;windows[key]=s
        with log.open(encoding='utf-8-sig') as stream:shots=[s for s in scan(stream,set(windows)) if s['eligible']]
        for shot in shots:
            s=windows[shot['window']];m=measure(shot,s);statuses[m['status']]+=1
            if m['status']!='measured':continue
            observed=s['observation'].get('kick_angles_degrees')
            if observed is None:continue
            view=vector(shot['fields']['view']);native=vector(shot['fields']['recoil'])
            # Hold actual view/muzzle/observed target fixed, replace only recoil.
            # This unexecuted ray uses a label oracle; never a runtime input.
            hypothetical=copy.deepcopy(shot)
            hypothetical['fields']['aim']=','.join(map(str,ray(view[1]+observed[1],view[0]+observed[0])))
            hypothetical['fields']['recoil']=','.join(map(str,observed))
            oracle=measure(hypothetical,s)
            entries.append(dict(seed=job['seed'],episode=task['episode']['id'],frame=s['observation']['identity']['frame'],
                shot=int(shot['fields']['shot']),observed_gun_frame=s['observation']['gun_frame'],
                native_pitch=native[0],observed_pitch=observed[0],pitch_delta=native[0]-observed[0],
                actual_error=m['post_recoil_degrees'],pre_recoil_error=m['pre_recoil_degrees'],
                delta_compensation_error=oracle['post_recoil_degrees'],signed_pitch_error=m['post_pitch_error'],
                selected_target_damage=m['selected_target_damage'],moving=m['native_horizontal_speed']>=10))
    assert len(entries)==noise['groups']['all']['shots'] and dict(statuses)==noise['native_statuses']
    def summary(values):
        mean=lambda key:sum(v[key] for v in values)/len(values) if values else None
        return dict(shots=len(values),battles=len({v['seed'] for v in values}),
            **{k:mean(k) for k in ('native_pitch','observed_pitch','pitch_delta','signed_pitch_error','actual_error','pre_recoil_error','delta_compensation_error')},
            pitch_delta_abs_mean=sum(abs(v['pitch_delta']) for v in values)/len(values) if values else None,
            delta_compensation_better_shots=sum(v['delta_compensation_error']<v['actual_error'] for v in values),
            selected_target_damage=sum(v['selected_target_damage'] for v in values))
    groups={'all':summary(entries)}
    for name,predicate in [('ordinary',lambda s:not s['episode'].endswith('-recovery')),('recovery',lambda s:s['episode'].endswith('-recovery')),
            ('moving',lambda s:s['moving']),('stationary',lambda s:not s['moving']),('damage',lambda s:s['selected_target_damage']),('no_damage',lambda s:not s['selected_target_damage'])]:
        groups[name]=summary([s for s in entries if predicate(s)])
    phases={str(phase):summary([s for s in entries if s['observed_gun_frame']==phase]) for phase in sorted({s['observed_gun_frame'] for s in entries})}
    for path,digest in pins.items():assert sha(Path(path))==digest,path
    report=dict(state='complete',version='combat_recoil_alignment_v1',groups=groups,observed_phases=phases,
        inputs=pins,script_sha256=sha(__file__),neural_execution=False,training_performed=False,promotion=False,
        scope='TRAIN only. Oracle one-step geometric effect of extra -(native recoil - observed recoil), holding actual view/muzzle/target fixed. Not a predicted hit, valid teacher, executed action, runtime hidden input, or closed-loop quality claim.')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(groups=groups,observed_phases=phases)))


if __name__=='__main__':main()
