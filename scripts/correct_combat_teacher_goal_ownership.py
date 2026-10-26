"""Derive explicit goal-stop ownership corrections; retain every original record."""
import argparse
import copy
import json
from pathlib import Path

from combat_native_teacher_registry import VERSION, load_index, validate_record
from combat_teacher_ownership import rules_ownership
from process_combat_architecture_pool import read, save, sha
from recover_combat_anchor_screen import first_life


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();repo=Path(__file__).resolve().parents[1];out=a.out.resolve()
    assert not out.exists();records=load_index(repo,a.index);out.mkdir(parents=True)
    snapshot=out/'source-index.json';snapshot.write_bytes(a.index.read_bytes())
    pins={str(snapshot):sha(snapshot),str(Path(__file__).resolve()):sha(__file__)}
    for name in ('combat_teacher_ownership.py','combat_native_teacher_registry.py'):
        p=repo/'scripts'/name;pins[str(p)]=sha(p)
    parents={}
    for corpus in read(snapshot)['corpora']:
        manifest=(repo/corpus['manifest']).resolve();pins[str(manifest)]=sha(manifest)
        for entry in read(manifest)['entries']:
            path=Path(entry['path']);parents[entry['id']]=(path,entry['sha256']);pins[str(path)]=entry['sha256']
    save(out/'protocol.json',dict(version='combat_teacher_active_ownership_correction_v2',
        scope='Post-hoc metadata bugfix;all65 original contexts retained,inputs and utilities unchanged. Old corpora and index snapshot remain evidence. Never ordinary PPO data or policy quality evidence.',
        exemption='Only a quiet rules row at last kill,terminal combat_goal_complete,next living frame bound to verified native objective alive0/pending0,matching server dispatch,empty damage indexes,no movement/attack/jump/impulse. Every other rules row remains active.',
        training_performed=False,promotion=False,parent_index_sha256=sha(snapshot)))
    cache={};entries=[];counts=dict(choice0=0,choice1=0,tie=0,ineligible=0)
    changes=[];passive_count=0
    for original in records:
        r=copy.deepcopy(original);evidence={}
        for mode in ('choice0','choice1'):
            directory=Path(r['outcomes'][mode]['data']).resolve();root=directory.parent
            if root not in cache:
                pool=read(root/'pool/report.json');request=read(root/'cuda-request.json');batch=read(root/'cuda-batch/report.json')
                assert pool['state']=='complete' and pool['source_unchanged'] and batch['state']=='complete'
                assert sha(root/'cuda-request.json')==batch['request_sha256']
                jobs=sorted(pool['jobs'],key=lambda j:j['plan_index'])
                assert pool['usable_captures']==len(jobs)==len(request['tasks'])==len(batch['tasks'])
                cache[root]=(jobs,request,batch)
                for p in (root/'pool/report.json',root/'cuda-request.json',root/'cuda-batch/report.json'):
                    pins[str(p)]=sha(p)
            jobs,request,batch=cache[root]
            index=next(i for i,t in enumerate(request['tasks']) if Path(t['out']).resolve()==directory)
            job,task,receipt=jobs[index],request['tasks'][index],batch['tasks'][index]
            assert job['seed']==r['seed'] and not job['error']
            native_path=Path(task['data'])/'report.json';native=read(native_path)
            assert sha(native_path)==task['native_report_sha256']==receipt['native_report_sha256']
            assert sha(directory/'report.json')==receipt['report_sha256']
            assert sha(directory/'cuda-verification.json')==receipt['cuda_verification_sha256']
            pins[str(native_path)]=sha(native_path)
            cap_path=Path(job['root'])/'report.json'
            assert str(cap_path) in native['source_sha256'] and sha(cap_path)==native['source_sha256'][str(cap_path)]
            cap=read(cap_path)['results'][0]
            step_paths=[Path(p) for p in native['source_sha256'] if p.endswith('steps.jsonl')]
            assert len(step_paths)==1;steps_path=step_paths[0]
            assert sha(steps_path)==native['source_sha256'][str(steps_path)]
            for p,digest in native['source_sha256'].items():
                if p.endswith(('steps.jsonl','server.log','goal-stop.json','manifest.json')) or p==str(cap_path):
                    assert sha(p)==digest;pins[p]=digest
            with steps_path.open(encoding='utf-8-sig') as stream:
                rows=first_life(cap,[json.loads(line) for line in stream])
            audit=rules_ownership(rows,cap['goal_stop'],cap['first_life']['end_frame'])
            old=r['outcomes'][mode]['native']['rules_combat_rows']
            assert old==audit['original_inclusive_rules_rows'],(r['map'],mode,old,audit)
            r['outcomes'][mode]['native']['rules_combat_rows']=audit['rules_combat_rows']
            r['outcomes'][mode]['native']['passive_goal_stop_rows']=len(audit['passive_goal_stop_frames'])
            evidence[mode]=dict(**audit,steps_path=str(steps_path),steps_sha256=sha(steps_path),
                                native_report=str(native_path),native_report_sha256=sha(native_path),
                                verified_goal_stop=cap['goal_stop'])
            passive_count+=len(audit['passive_goal_stop_frames'])
        reasons=[mode+':rules_owned_continuation' for mode in ('choice0','choice1') if r['outcomes'][mode]['native']['rules_combat_rows']]
        delta=r['target']['utility_delta'];preference=None if abs(delta)<r['target']['tie_margin'] else int(delta>0)
        r['target'].update(eligible_for_teacher_supervision=not reasons,ineligible_reasons=reasons,
                           preference=preference if not reasons else None)
        parent_path,parent_sha=parents[r['id']]
        r['ownership_admission']=dict(version='combat_teacher_active_combat_ownership_v2',
            parent_record=str(parent_path),parent_record_sha256=parent_sha,evidence=evidence,
            posthoc_scope_correction=True,inputs_unchanged=True,utility_unchanged=True)
        assert r['input']==original['input'] and r['target']['utility_delta']==original['target']['utility_delta']
        validate_record(r)
        category='ineligible' if reasons else 'tie' if preference is None else 'choice'+str(preference)
        counts[category]+=1
        if r['target']!=original['target']:
            changes.append(dict(id=r['id'],map=r['map'],seed=r['seed'],frame=r['frame'],
                                previous_target=original['target'],corrected_target=r['target']))
        path=out/(r['id']+'.json');save(path,r)
        entries.append(dict(id=r['id'],map=r['map'],seed=r['seed'],path=str(path),sha256=sha(path),
                            preference=r['target']['preference'],teacher_eligible=not reasons))
    for path,digest in pins.items():assert sha(path)==digest,path
    save(out/'manifest.json',dict(version=VERSION,state='complete',entries=entries,counts=counts,inputs=pins,
        ownership_version='combat_teacher_active_combat_ownership_v2',posthoc_scope_correction=True,
        ordinary_ppo_eligible=False,diagnostic_only=True,training_performed=False,promotion=False))
    save(out/'complete.json',dict(state='complete',records=len(entries),counts=counts,
        corrected_eligibility_records=len(changes),passive_goal_stop_rows=passive_count,changes=changes,
        manifest_sha256=sha(out/'manifest.json'),training_performed=False,promotion=False))
    print(json.dumps(dict(records=len(entries),counts=counts,corrected_records=len(changes),passive_rows=passive_count)))


if __name__=='__main__':main()
