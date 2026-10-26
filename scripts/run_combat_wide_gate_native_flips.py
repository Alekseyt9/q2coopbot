"""Paired native diagnostics of first TRAIN choices changed by gate fitting."""
import argparse,copy,json,os,shutil,subprocess,sys
from pathlib import Path
from process_combat_architecture_pool import read,save,sha
from evaluate_registered_combat_models import pool_command
from finish_combat_anchor_probe import first_diff

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('batch','audit','out'):ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args();repo=Path(__file__).resolve().parents[1];root=a.out.resolve();assert not root.exists()
    batch=a.batch.resolve();audit=read(a.audit/'report.json');assert audit['diagnostic_only'] and not audit['training_performed']
    candidate=next(v for p,v in audit['batches'].items() if Path(p).resolve()==batch)
    selected={}
    for e in candidate['events']:
        if e['choice']!=e['counterfactual_choice_after_fit'] and e['seed'] not in selected:selected[e['seed']]=e
    assert len(selected)==5,'This frozen mechanism probe requires all five second-batch first flips'
    original=read(batch/'train-plan.json');protocol=read(batch/'protocol.json');source=Path(protocol['source'])
    models={'before':source/'weights.json','after':batch/'fit/weights.json'}
    root.mkdir(parents=True);pins={str(p.resolve()):sha(p) for p in (a.audit/'report.json',batch/'train-plan.json',*models.values())}
    os.environ['GOCACHE']=str(repo/'workspace/build/gocache');os.environ['Q2COOPBOT_DIAGNOSTIC_ENGINE_ROOT']=str(repo/'workspace/artifacts/combat-objective-engine-v2-20261025')
    def run(label,cmd):
        save(root/'progress.json',dict(stage=label))
        with (root/(label+'.log')).open('w',encoding='utf-8') as log:
            p=subprocess.run(list(map(str,cmd)),cwd=repo,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert p.returncode==0,str(root/(label+'.log'))
    try:
        planner=root/'q2episode.exe';exporter=root/'q2ppo-data.exe'
        run('build-planner',['go','build','-o',planner,'./cmd/q2episode']);run('build-exporter',['go','build','-o',exporter,'./cmd/q2ppo-data'])
        plans=[]
        for label,model in models.items():
            plan=copy.deepcopy(original);plan['tasks']=[t for t in plan['tasks'] if t['seeds'][0] in selected]
            assert len(plan['tasks'])==5
            plan.update(model_path=str(model),model_sha256=sha(model),output_root=str(root/('captures-'+label)))
            path=root/(label+'-plan.json');save(path,plan);run('verify-'+label,[planner,'--verify-plan',path,'--root',repo]);plans.append(path);pins[str(path)]=sha(path)
        save(root/'protocol.json',dict(state='frozen',version='combat_learned_wide_native_flip_probe_v1',selected=list(selected.values()),inputs=pins,
            selection='All five second-batch TRAIN cases whose first recorded gate decision changes under fitted head on original observed states; no outcome or damage filtering.',
            native_cases=10,cuda_cases=10,max_instances=16,timescale=2,diagnostic_only=True,ordinary_training_eligible=False,training_performed=False,promotion=False,
            acceptance='Same observed state and sent command prefix before predicted first flip. Same current observation at flip, different learned wide choice; inspect actual sent command. No ID normalization or hidden root selection.',
            scope='Reused TRAIN mechanism probe with ordinary before/after policies, no forced action. Not fresh quality or architecture acceptance.'))
        run('collect',[shutil.which('pwsh'),'-NoProfile','-Command',pool_command(repo,plans,16,54300,root/'pool')])
        pool=read(root/'pool/report.json');assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==len(pool['jobs'])==10
        tasks=[];pairs={};refs=[]
        for i,j in enumerate(sorted(pool['jobs'],key=lambda j:(j['plan_index'],j['task_index']))):
            assert not j['error'];label=('before','after')[j['plan_index']];model=models[label]
            capture=read(Path(j['root'])/'report.json')['results'][0];native=root/f'export-{i:02d}-native';cuda=root/f'export-{i:02d}-cuda'
            run('export-'+str(i),[exporter,'--batch',j['root'],'--model',model,'--cuda-only','--out',native])
            tasks.append(dict(model=str(model),model_sha256=sha(model),data=str(native),out=str(cuda),native_report_sha256=sha(native/'report.json')))
            rows=[json.loads(l) for l in (Path(capture['root'])/'dataset/steps.jsonl').open(encoding='utf-8-sig')]
            active=[s for s in rows if s['observation']['identity']['life']==1 and s['observation']['health']>0 and s['observation']['identity']['frame']<=capture['first_life']['end_frame']]
            pairs.setdefault(j['seed'],{})[label]=(capture,active);refs.append(dict(label=label,seed=j['seed'],raw=str(cuda)))
        save(root/'cuda-request.json',dict(version='combat_cuda_finalize_batch_v1',compact_finalized_streams=True,tasks=tasks))
        run('cuda',[sys.executable,repo/'scripts/finalize_combat_cuda_batch.py','--request',root/'cuda-request.json','--receipt-root',root/'cuda-batch'])
        assert read(root/'cuda-batch/report.json')['state']=='complete'
        # Publish only derived diagnostic references; preserve numerical proofs.
        reference=root/'diagnostic-references';reference.mkdir()
        for i,r in enumerate(refs):
            raw=Path(r['raw']);meta=read(raw/'report.json');dest=reference/f'case-{i:02d}';dest.mkdir()
            for f in ('rollout.jsonl','sequence.jsonl','cuda-verification.json'):os.link(raw/f,dest/f)
            meta.update(version='combat_diagnostic_reference_rollout_v1',diagnostic_only=True,ordinary_training_eligible=False,ppo_training_ready=False,raw_report=str(raw/'report.json'),raw_report_sha256=sha(raw/'report.json'),purpose='reused_train_learned_choice_flip_mechanism')
            save(dest/'report.json',meta);r.update(data=str(dest),diagnostic_only=True)
        save(root/'diagnostic-index.json',dict(state='complete',entries=refs,ordinary_training_eligible=False,training_performed=False))
        reports=[]
        for seed,branches in pairs.items():
            before,aa=branches['before'];after,bb=branches['after'];frame=selected[seed]['frame']
            assert before['generated_fixture']==after['generated_fixture']
            prefix=lambda rr:[dict(observation=s['observation'],sent_command=s['sent_command'],owner=s['owner']) for s in rr if s['observation']['identity']['frame']<frame]
            assert first_diff(prefix(aa),prefix(bb)) is None,dict(seed=seed,error='native observed prefix differs')
            ar=next(s for s in aa if s['observation']['identity']['frame']==frame);br=next(s for s in bb if s['observation']['identity']['frame']==frame)
            assert first_diff(ar['observation'],br['observation']) is None,dict(seed=seed,error='root observation differs')
            assert ar['sample']['wide_aim']['choice']==selected[seed]['choice'] and br['sample']['wide_aim']['choice']==selected[seed]['counterfactual_choice_after_fit']
            metrics={}
            for label,(cap,rr) in branches.items():
                d=cap['first_life']['damage'];metrics[label]=dict(goal=bool(cap['goal_stop']),death=bool(cap['death_stop'] or cap['first_life']['minimum_health']<=0),kills=sum(v['monster_kills'] for v in d['by_mod']),dealt=sum(v['monster_health_damage'] for v in d['by_mod']),received=d['received_health_damage'],rules_rows=sum(s['owner']=='rules' for s in rr),attack_commands=sum(bool(s['sent_command']['Buttons']&1) for s in rr))
            reports.append(dict(seed=seed,map=before['first_life']['map'],root_frame=frame,prefix_exact=True,root_observation_exact=True,choice_before=ar['sample']['wide_aim']['choice'],choice_after=br['sample']['wide_aim']['choice'],sent_before=ar['sent_command'],sent_after=br['sent_command'],changed_command_fields=[k for k,v in ar['sent_command'].items() if v!=br['sent_command'][k]],metrics=metrics))
        for p,h in pins.items():assert sha(p)==h
        save(root/'results.json',dict(state='complete',diagnostic_only=True,cases=reports,training_performed=False,promotion=False,scope='Five reused TRAIN worlds selected by first changed gate choice, not outcomes. Attack command counts are not shots or accuracy; rules ownership explicit. Not independent quality.'))
        save(root/'complete.json',dict(state='complete',native_cases=10,cuda_cases=10,diagnostic_only=True,results_sha256=sha(root/'results.json'),diagnostic_index_sha256=sha(root/'diagnostic-index.json'),training_performed=False,promotion=False))
        save(root/'progress.json',dict(stage='complete'));print(root/'results.json')
    except BaseException as e:save(root/'progress.json',dict(stage='failed',error=str(e)));raise

if __name__=='__main__':main()
