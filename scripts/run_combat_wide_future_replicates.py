"""Four additional fixed RNG continuations for every prior diagnostic context."""
import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from combat_continuation_probe import VERSION as FORK_VERSION
from combat_native_teacher_registry import load_index
from combat_teacher_end_receipt import admitted_terminal_receipt_v3
from combat_teacher_ownership import rules_ownership
from combat_wide_aim import PROBE_VERSION
from evaluate_registered_combat_models import pool_command
from finish_combat_anchor_probe import first_diff
from process_combat_architecture_pool import read,save,sha
from recover_combat_anchor_screen import first_life


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('index','source','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--prior-protocol',type=Path,action='append',required=True)
    a=ap.parse_args();repo=Path(__file__).resolve().parents[1];out=a.out.resolve()
    assert not out.exists()
    records=load_index(repo,a.index)
    previous=[(p.resolve(),read(p)) for p in a.prior_protocol]
    ids=[c['record_id'] for _,p in previous for c in p['contexts']]
    assert len(ids)==len(set(ids))==8
    selected=[next(r for r in records if r['id']==rid) for rid in ids]
    expected_cases=12*len(selected)
    futures=('original','forkC','forkD','forkE','forkF')
    source=a.source.resolve();source_model=read(source);assert all(r['source_model_sha256']==sha(source) for r in selected)
    assert not any(k.startswith('diagnostic_') for k in source_model)
    out.mkdir(parents=True);(out/'models').mkdir();(out/'frozen-scripts').mkdir();(out/'frozen-go').mkdir()
    pins={str(source):sha(source),str(a.index.resolve()):sha(a.index)}
    historical_collector_snapshots=[]
    for path,prior in previous:
        pins[str(path)]=sha(path)
        for p,h in prior['inputs'].items():
            if sha(p)!=h:
                assert Path(p).name=='run_combat_wide_future_stability.py'
                frozen=path.parent/'frozen-scripts'/Path(p).name
                snapshot=path.parent/'collector-source-after-run.py'
                assert sha(frozen)==sha(snapshot)==h
                pins[str(frozen)]=h;pins[str(snapshot)]=h
                historical_collector_snapshots.append(dict(original_path=p,original_sha256=h,frozen=str(frozen),current_sha256=sha(p)))
            else:pins[p]=h
    for corpus in read(a.index)['corpora']:
        manifest=(repo/corpus['manifest']).resolve();pins[str(manifest)]=sha(manifest)
        for entry in read(manifest)['entries']:pins[entry['path']]=entry['sha256']
    os.environ['GOCACHE']=str(repo/'workspace/build/gocache')
    os.environ['Q2COOPBOT_DIAGNOSTIC_ENGINE_ROOT']=str(repo/'workspace/artifacts/combat-objective-engine-v2-20261025')

    def run(label,command):
        save(out/'progress.json',dict(stage=label))
        with (out/(label+'.log')).open('w',encoding='utf-8') as stream:
            process=subprocess.run(list(map(str,command)),cwd=repo,stdout=stream,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert process.returncode==0,str(out/(label+'.log'))

    try:
        planner,exporter=out/'q2episode.exe',out/'q2ppo-data.exe'
        run('build-planner',['go','build','-o',planner,'./cmd/q2episode'])
        run('build-exporter',['go','build','-o',exporter,'./cmd/q2ppo-data'])
        assignments=[];plans=[];contexts=[]
        for r in selected:
            cid=r['id'][:12];old=Path(r['outcomes']['choice0']['data']).parent
            original_path=old/f"{r['seed']}-choice0-plan.json";original=read(original_path);pins[str(original_path)]=sha(original_path)
            assert len(original['tasks'])==1 and original['tasks'][0]['seeds']==[r['seed']]
            contexts.append(dict(id=cid,map=r['map'],seed=r['seed'],frame=r['frame'],record_id=r['id'],original_root=str(old)))
            for future,seed in (('forkC',410003),('forkD',530017),('forkE',670019),('forkF',790021)):
                for mode in ('baseline','choice0','choice1'):
                    value=copy.deepcopy(source_model)
                    if seed is not None:value['diagnostic_continuation_probe']=dict(version=FORK_VERSION,fork_frame=r['frame']+1,seed=seed)
                    if mode!='baseline':value['diagnostic_wide_gate_probe']=dict(version=PROBE_VERSION,frame=r['frame'],choice=int(mode[-1]))
                    core=copy.deepcopy(value);core.pop('diagnostic_continuation_probe',None);core.pop('diagnostic_wide_gate_probe',None)
                    assert core==source_model
                    label=cid+'-'+future+'-'+mode;model=out/'models'/(label+'.json');save(model,value)
                    plan=copy.deepcopy(original);plan.update(model_path=str(model),model_sha256=sha(model),output_root=str(out/('captures-'+label)))
                    path=out/(label+'-plan.json');save(path,plan)
                    run('verify-'+label,[planner,'--verify-plan',path,'--root',repo]);plans.append(path)
                    assignments.append(dict(plan_index=len(plans)-1,context_id=cid,seed=r['seed'],frame=r['frame'],future=future,future_seed=seed,mode=mode,model=str(model),model_sha256=sha(model)))
                    pins[str(path)]=sha(path);pins[str(model)]=sha(model)
        helpers=('run_combat_wide_future_replicates.py','combat_teacher_end_receipt.py','combat_continuation_probe.py','combat_wide_aim.py',
            'combat_joint_aim_policy.py','finalize_combat_cuda_rollout.py','finalize_combat_cuda_batch.py',
            'combat_teacher_native_receipt.py','combat_teacher_movement_receipt.py','combat_teacher_ownership.py')
        for name in helpers:
            path=repo/'scripts'/name;pins[str(path)]=sha(path);shutil.copyfile(path,out/'frozen-scripts'/name)
        for name in ('continuation_probe.go','wide_gate_probe.go','wide_aim.go','ppo.go','aim_candidates.go'):
            path=repo/'internal/policy'/name;pins[str(path)]=sha(path);shutil.copyfile(path,out/'frozen-go'/name)
        save(out/'protocol.json',dict(version='combat_wide_future_replicates_v1',contexts=contexts,assignments=assignments,inputs=pins,
            native_cases=expected_cases,cuda_cases=expected_cases,max_instances=16,timescale=2,
            selection='All eight previous SHA-selected contexts; no selection by preference,success or stability',
            prior_protocols=[str(p) for p,_ in previous],historical_collector_snapshots=historical_collector_snapshots,
            continuation='Same weights/world/root clamp, four additional fixed suffix seeds at root+1. Historical original triples reused with SHA-bound CUDA receipts. No new fitting.',
            acceptance='Exact admitted prefix and current observation;root samples/actions invariant across future modes. Matching clamp reproduces baseline full admitted life and reconciled terminal per future mode. Fork reset receipts/Gaussian/finalfire/category/wideCDF verified on CUDA.',
            scope='Reused TRAIN contexts; four additional continuations plus reused original per choice. Prior forkA/B remain separate for pooled seven-stream analysis. Not precise expected Q or policy quality; retain failures/hybrids,never ordinary PPO.',
            training_performed=False,diagnostic_only=True,ordinary_ppo_eligible=False,promotion=False))
        run('collect',[shutil.which('pwsh'),'-NoProfile','-Command',pool_command(repo,plans,16,54800,out/'pool')])
        pool=read(out/'pool/report.json');assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==len(pool['jobs'])==expected_cases
        groups={};native_dirs={};tasks=[];exports=[];terminal_proofs=[]
        for i,job in enumerate(sorted(pool['jobs'],key=lambda j:j['plan_index'])):
            assert not job['error'];assignment=assignments[job['plan_index']]
            cap=read(Path(job['root'])/'report.json')['results'][0]
            with (Path(cap['root'])/'dataset/steps.jsonl').open(encoding='utf-8-sig') as stream:
                rows=first_life(cap,[json.loads(line) for line in stream])
            groups[(assignment['context_id'],assignment['future'],assignment['mode'])]=(cap,rows)
            native,cuda=out/f'export-{i:02d}-native',out/f'export-{i:02d}-cuda'
            native_dirs[(assignment['context_id'],assignment['future'],assignment['mode'])]=native
            run('export-'+str(i),[exporter,'--batch',job['root'],'--model',assignment['model'],'--cuda-only','--out',native])
            tasks.append(dict(model=assignment['model'],model_sha256=assignment['model_sha256'],data=str(native),out=str(cuda),native_report_sha256=sha(native/'report.json')))
            exports.append(dict(**assignment,data=str(cuda)))
        save(out/'cuda-request.json',dict(version='combat_cuda_finalize_batch_v1',compact_finalized_streams=True,tasks=tasks))
        run('cuda',[sys.executable,repo/'scripts/finalize_combat_cuda_batch.py','--request',out/'cuda-request.json','--receipt-root',out/'cuda-batch'])
        batch=read(out/'cuda-batch/report.json')
        assert batch['state']=='complete' and batch['device']=='cuda' and len(batch['tasks'])==expected_cases
        assert batch['request_sha256']==sha(out/'cuda-request.json')
        for i,(task,audit) in enumerate(zip(tasks,batch['tasks'])):
            assert audit['task']==i and task['model_sha256']==audit['model_sha256'] and task['native_report_sha256']==audit['native_report_sha256']
            assert task['out']==audit['out']
            native,directory=Path(task['data']),Path(task['out'])
            for path,h in ((native/'report.json',task['native_report_sha256']),(directory/'report.json',audit['report_sha256']),
                           (directory/'rollout.jsonl',audit['rollout_sha256']),(directory/'cuda-verification.json',audit['cuda_verification_sha256'])):
                assert sha(path)==h;pins[str(path)]=h
            for path,h in read(native/'report.json')['source_sha256'].items():assert sha(path)==h; pins[path]=h

        # Keep every original branch; reuse existing proven captures without new games.
        for prior_path,prior in previous:
            base=prior_path.parent;request=read(base/'cuda-request.json');batch=read(base/'cuda-batch/report.json')
            assert batch['state']=='complete' and batch['request_sha256']==sha(base/'cuda-request.json')
            for name in ('cuda-request.json','cuda-batch/report.json','pool/report.json'):
                path=base/name;pins[str(path)]=sha(path)
            old_jobs={j['plan_index']:j for j in read(base/'pool/report.json')['jobs']}
            for item in prior['assignments']:
                if item['future']!='original':continue
                i=item['plan_index'];task=request['tasks'][i];audit=batch['tasks'][i]
                assert audit['task']==i and task['model']==item['model'] and audit['model_sha256']==item['model_sha256']
                directory=Path(task['out']);native=Path(task['data'])
                for path,h in ((native/'report.json',task['native_report_sha256']),(directory/'report.json',audit['report_sha256']),
                               (directory/'rollout.jsonl',audit['rollout_sha256']),(directory/'cuda-verification.json',audit['cuda_verification_sha256'])):
                    assert sha(path)==h;pins[str(path)]=h
                for path,h in read(native/'report.json')['source_sha256'].items():assert sha(path)==h; pins[path]=h
                path=Path(old_jobs[i]['root'])/'report.json';pins[str(path)]=sha(path);cap=read(path)['results'][0]
                path=Path(cap['root'])/'dataset/steps.jsonl';pins[str(path)]=sha(path)
                with path.open(encoding='utf-8-sig') as stream:rows=first_life(cap,[json.loads(line) for line in stream])
                key=(item['context_id'],item['future'],item['mode']);assert key not in groups
                groups[key]=(cap,rows);native_dirs[key]=native;exports.append(dict(**item,data=str(directory),reused=True))
        project=lambda rows:[dict(observation=s['observation'],sent_command=s['sent_command'],owner=s['owner']) for s in rows]
        def sample(step):
            value=copy.deepcopy(step.get('sample'));assert value is not None;value.pop('version',None);return value
        results=[]
        for ctx in contexts:
            cid,frame=ctx['id'],ctx['frame'];reference=groups[(cid,'original','baseline')]
            prefixes=[];outcomes=[]
            for future in futures:
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
                            terminal_proofs.append(dict(context_id=cid,future=future,baseline=lp,matching=rp))
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
                for future in futures[1:]:
                    cap,rows=groups[(cid,future,mode)]
                    candidates=[s for s in rows if s['observation']['identity']['frame']==frame+1 and s.get('sample',{}).get('continuation_probe')]
                    noises.append(candidates[0]['sample']['continuation_probe']['gaussian_noise'] if candidates else None)
                if all(n is not None for n in noises):assert len({str(n) for n in noises})==len(noises),'Future Gaussian streams did not change'
                stream_checks.append(dict(mode=mode,all_forks_observed=all(n is not None for n in noises),streams_changed=len({str(n) for n in noises})==len(noises) if all(n is not None for n in noises) else None))
            for future in futures:
                zero=next(v for v in outcomes if v['future']==future and v['mode']=='choice0')
                one=next(v for v in outcomes if v['future']==future and v['mode']=='choice1')
                eligible=zero['ownership']['rules_combat_rows']==one['ownership']['rules_combat_rows']==0
                delta=one['utility']-zero['utility']
                preference=None if not eligible or abs(delta)<.05 else int(delta>0)
                differences.append(dict(future=future,delta=delta,eligible=eligible,preference=preference))
            signs={v['preference'] for v in differences if v['preference'] is not None}
            results.append(dict(context=ctx,checks=prefixes,historical_root_exact=True,future_stream_checks=stream_checks,outcomes=outcomes,differences=differences,preference_flip=signs=={0,1}))
        for p,digest in pins.items():assert sha(p)==digest,p
        save(out/'verification-inputs.json',pins)
        save(out/'terminal-proofs.json',dict(entries=terminal_proofs))
        save(out/'exports.json',dict(entries=exports,diagnostic_only=True,ordinary_ppo_eligible=False))
        save(out/'results.json',dict(state='complete',cases=results,training_performed=False,promotion=False,diagnostic_only=True))
        save(out/'complete.json',dict(state='complete',native_cases=expected_cases,cuda_cases=expected_cases,contexts=len(contexts),
            results_sha256=sha(out/'results.json'),preference_flip_contexts=sum(v['preference_flip'] for v in results),
            training_performed=False,promotion=False,ordinary_ppo_eligible=False))
        save(out/'progress.json',dict(stage='complete'));print(out/'complete.json')
    except BaseException as error:
        save(out/'progress.json',dict(stage='failed',error=str(error)));raise


if __name__=='__main__':main()
