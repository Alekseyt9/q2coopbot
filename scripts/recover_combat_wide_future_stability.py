"""Read-only post-hoc closure of every completed wide continuation branch.

The original failed verdict is immutable. Only optional EOF death-stop metadata
is replaced by exact native lethal-event evidence; no game or CUDA run repeated.
"""
import argparse
import copy
import json
from pathlib import Path

from combat_native_teacher_registry import load_index
from combat_teacher_end_receipt import admitted_terminal_receipt_v3
from combat_teacher_ownership import rules_ownership
from finish_combat_anchor_probe import first_diff
from process_combat_architecture_pool import read,save,sha
from recover_combat_anchor_screen import first_life


def recover(original,out):
    original,out=original.resolve(),out.resolve()
    repo=Path(__file__).resolve().parents[1]
    assert not out.exists() and read(original/'progress.json')['stage']=='failed'
    protocol=read(original/'protocol.json');pins=dict(protocol['inputs'])
    def pin(path,expected=None):
        path=Path(path);digest=sha(path)
        if expected is not None:assert digest==expected,str(path)
        if str(path) in pins:assert pins[str(path)]==digest,str(path)
        pins[str(path)]=digest
    for path,digest in list(pins.items()):assert sha(path)==digest,path
    for name in ('protocol.json','progress.json','pool/report.json','cuda-request.json','cuda-batch/report.json'):pin(original/name)
    for name in ('recover_combat_wide_future_stability.py','combat_teacher_end_receipt.py','combat_teacher_native_receipt.py','combat_teacher_movement_receipt.py','recover_combat_anchor_screen.py','combat_teacher_ownership.py','finish_combat_anchor_probe.py','combat_native_teacher_registry.py','process_combat_architecture_pool.py'):pin(repo/'scripts'/name)
    pool=read(original/'pool/report.json');assignments=protocol['assignments'];contexts=protocol['contexts']
    request=read(original/'cuda-request.json');batch=read(original/'cuda-batch/report.json')
    count=protocol['native_cases']
    assert count==protocol['cuda_cases']==9*len(contexts)==len(assignments)==len(pool['jobs'])==pool['usable_captures']==len(request['tasks'])==len(batch['tasks'])
    assert pool['state']=='complete' and pool['source_unchanged']
    assert batch['state']=='complete' and batch['device']=='cuda' and batch['request_sha256']==sha(original/'cuda-request.json')
    assert [j['plan_index'] for j in sorted(pool['jobs'],key=lambda j:j['plan_index'])]==list(range(count))
    selected=load_index(repo,repo/'scripts/scenarios/combat-teachers/index.json')
    assert {c['record_id'] for c in contexts}.issubset({r['id'] for r in selected})
    groups={};native_dirs={};exports=[];terminal_proofs=[]
    for i,job in enumerate(sorted(pool['jobs'],key=lambda j:j['plan_index'])):
        assignment=assignments[i];task=request['tasks'][i];audit=batch['tasks'][i]
        assert assignment['plan_index']==audit['task']==i and not job['error'] and job['seed']==assignment['seed']
        assert task['model']==assignment['model'] and task['model_sha256']==assignment['model_sha256']==audit['model_sha256']
        pin(task['model'],task['model_sha256']);native=Path(task['data']);directory=Path(task['out'])
        assert str(directory)==audit['out'] and task['native_report_sha256']==audit['native_report_sha256']
        pin(native/'report.json',task['native_report_sha256'])
        meta=read(native/'report.json')
        for path,digest in meta['source_sha256'].items():pin(path,digest)
        pin(directory/'report.json',audit['report_sha256'])
        pin(directory/'rollout.jsonl',audit['rollout_sha256'])
        pin(directory/'cuda-verification.json',audit['cuda_verification_sha256'])
        proof=read(directory/'cuda-verification.json');assert proof['state']=='passed' and proof['device']=='cuda' and proof['model_sha256']==assignment['model_sha256']
        assert proof['wide_source_transform_and_cdf_verified']
        cap_report=Path(job['root'])/'report.json';pin(cap_report);cap_meta=read(cap_report)
        assert cap_meta['capture_complete'] and cap_meta['provenance_valid'] and len(cap_meta['results'])==1
        cap=cap_meta['results'][0]
        assert cap['capture_valid'] and cap['dispatch_valid'] and cap['seed_confirmed']
        path=Path(cap['root'])/'dataset/steps.jsonl';pin(path);pin(Path(cap['root'])/'bot.jsonl')
        with path.open(encoding='utf-8-sig') as stream:rows=first_life(cap,[json.loads(line) for line in stream])
        key=(assignment['context_id'],assignment['future'],assignment['mode']);assert key not in groups
        groups[key]=(cap,rows);native_dirs[key]=native
        exports.append(dict(**assignment,data=str(directory)))
    out.mkdir(parents=True)
    save(out/'protocol.json',dict(version='combat_wide_future_native_end_posthoc_v1',original_root=str(original),original_protocol_passed=False,
        posthoc_scope_correction=True,includes_all_cases=True,native_cases=count,cuda_cases=count,
        correction='Canonical death_stop requires exact pinned native terminal and lethal server event; reject contradictory existing stop tag. All trajectory, first-life and historical-root checks unchanged.',
        inputs=pins,training_performed=False,promotion=False,ordinary_ppo_eligible=False))
    project=lambda rows:[dict(observation=s['observation'],sent_command=s['sent_command'],owner=s['owner']) for s in rows]
    def sample(step):
        value=copy.deepcopy(step.get('sample'));assert value is not None;value.pop('version',None);return value
    results=[]
    for ctx in contexts:
        cid,frame=ctx['id'],ctx['frame'];reference=groups[(cid,'original','baseline')]
        prefixes=[];outcomes=[]
        for future in ('original','forkA','forkB'):
            bc,bb=groups[(cid,future,'baseline')]
            root=next(s for s in bb if s['observation']['identity']['frame']==frame)
            original=next(s for s in reference[1] if s['observation']['identity']['frame']==frame)
            assert first_diff(root['observation'],original['observation']) is None
            assert first_diff(sample(root),sample(original)) is None,'Future RNG changed baseline root'
            choice=root['sample']['wide_aim']['choice']
            for mode in ('baseline','choice0','choice1'):
                cap,rows=groups[(cid,future,mode)]
                assert cap['generated_fixture']==bc['generated_fixture']==reference[0]['generated_fixture']
                prefix=lambda rr:[s for s in project(rr) if s['observation']['identity']['frame']<frame]
                assert first_diff(prefix(reference[1]),prefix(rows)) is None,'Future/clamp changed prefix'
                current=next(s for s in rows if s['observation']['identity']['frame']==frame)
                assert first_diff(current['observation'],root['observation']) is None
                nofork=next(s for s in groups[(cid,'original',mode)][1] if s['observation']['identity']['frame']==frame)
                assert first_diff(sample(current),sample(nofork)) is None,'Future RNG changed intervened root'
                assert current['sent_command']==nofork['sent_command']
                if mode!='baseline':
                    assert current['sample']['wide_aim']['choice']==int(mode[-1])
                    if int(mode[-1])==choice:
                        assert first_diff(project(bb),project(rows)) is None,'Matching clamp changed admitted trajectory'
                        bound=bb[0]['observation']['identity']['frame']
                        left,lp=admitted_terminal_receipt_v3(bc,bound,native_dirs[(cid,future,'baseline')]);right,rp=admitted_terminal_receipt_v3(cap,bound,native_dirs[(cid,future,mode)])
                        terminal_proofs.append(dict(context_id=cid,future=future,mode=mode,baseline=lp,matching=rp,raw_stop_difference=first_diff(bc.get('death_stop'),cap.get('death_stop'))))
                        assert first_diff(left,right) is None,'Matching clamp changed admitted terminal'
                ownership=rules_ownership(rows,cap['goal_stop'],cap['first_life']['end_frame'])
                entry=next(e for e in exports if (e['context_id'],e['future'],e['mode'])==(cid,future,mode))
                directory=Path(entry['data']);meta=read(directory/'report.json');proof=read(directory/'cuda-verification.json')
                assert proof['state']=='passed' and proof['device']=='cuda'
                if future!='original':
                    assert meta['diagnostic_only'] and not meta['ordinary_training_eligible']
                    assert proof['diagnostic_continuation']['fire_verified']
                with (directory/'rollout.jsonl').open(encoding='utf-8-sig') as stream:
                    rr=[r for r in map(json.loads,stream) if r['frame']>=frame]
                assert rr
                utility=sum(.99**(r['frame']-frame)*r['reward'] for r in rr)
                damage=cap['first_life']['damage']
                outcomes.append(dict(future=future,mode=mode,utility=utility,goal=bool(cap['goal_stop']),
                    death=bool(cap['death_stop'] or cap['first_life']['minimum_health']<=0),
                    kills=sum(v['monster_kills'] for v in damage['by_mod']),dealt=sum(v['monster_health_damage'] for v in damage['by_mod']),
                    received=damage['received_health_damage'],ownership=ownership,
                    fork_observed=proof.get('diagnostic_continuation',{}).get('observed'),data=str(directory)))
            prefixes.append(dict(future=future,root_and_prefix_exact=True,matching_clamp_life_terminal_exact=True))
        differences=[]
        historical=next(r for r in selected if r['id']==ctx['record_id'])
        original_entry=next(e for e in exports if (e['context_id'],e['future'],e['mode'])==(cid,'original','baseline'))
        with (Path(original_entry['data'])/'rollout.jsonl').open(encoding='utf-8-sig') as stream:
            historical_root=next(r for r in map(json.loads,stream) if r['frame']==frame)
        assert historical_root['features']==historical['input']['features'],'Historical root features changed'
        assert historical_root['sample']['wide_aim']['source_latent']==historical['input']['source_latent'],'Historical source event changed'
        stream_checks=[]
        for mode in ('baseline','choice0','choice1'):
            noises=[]
            for future in ('forkA','forkB'):
                cap,rows=groups[(cid,future,mode)]
                candidates=[s for s in rows if s['observation']['identity']['frame']==frame+1 and s.get('sample',{}).get('continuation_probe')]
                noises.append(candidates[0]['sample']['continuation_probe']['gaussian_noise'] if candidates else None)
            if all(n is not None for n in noises):assert noises[0]!=noises[1],'Future Gaussian streams did not change'
            stream_checks.append(dict(mode=mode,both_forks_observed=all(n is not None for n in noises),streams_changed=noises[0]!=noises[1] if all(n is not None for n in noises) else None))
        for future in ('original','forkA','forkB'):
            zero=next(v for v in outcomes if v['future']==future and v['mode']=='choice0')
            one=next(v for v in outcomes if v['future']==future and v['mode']=='choice1')
            eligible=zero['ownership']['rules_combat_rows']==one['ownership']['rules_combat_rows']==0
            delta=one['utility']-zero['utility']
            preference=None if not eligible or abs(delta)<.05 else int(delta>0)
            differences.append(dict(future=future,delta=delta,eligible=eligible,preference=preference))
        signs={v['preference'] for v in differences if v['preference'] is not None}
        results.append(dict(context=ctx,checks=prefixes,historical_root_exact=True,future_stream_checks=stream_checks,outcomes=outcomes,differences=differences,preference_flip=signs=={0,1}))
    for path,digest in pins.items():assert sha(path)==digest,path
    save(out/'terminal-proofs.json',dict(entries=terminal_proofs))
    save(out/'exports.json',dict(entries=exports,diagnostic_only=True,ordinary_ppo_eligible=False))
    save(out/'results.json',dict(state='complete',cases=results,original_protocol_passed=False,posthoc_scope_correction=True,training_performed=False,promotion=False,diagnostic_only=True))
    save(out/'complete.json',dict(state='complete',native_cases=count,cuda_cases=count,contexts=len(contexts),original_protocol_passed=False,
        results_sha256=sha(out/'results.json'),terminal_proofs_sha256=sha(out/'terminal-proofs.json'),
        preference_flip_contexts=sum(v['preference_flip'] for v in results),training_performed=False,promotion=False,ordinary_ppo_eligible=False))
    print(json.dumps(read(out/'complete.json')),flush=True)
    for value in results:print(value['context']['map'],value['differences'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();recover(args.root,args.out)
