"""Paired native controls for one-frame coupled aim/fire teacher roots."""
import argparse,copy,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from process_combat_architecture_pool import read,save,sha
from evaluate_registered_combat_models import pool_command
from recover_combat_anchor_screen import first_life,terminal_receipt
from finish_combat_anchor_probe import first_diff
from combat_wide_aim import PROBE_VERSION
from combat_teacher_root_selection import choose_root_frame
from combat_native_teacher_registry import load_index
from combat_teacher_movement_receipt import admitted_terminal_receipt_v2 as admitted_terminal_receipt
from combat_teacher_ownership import rules_ownership

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--reference',type=Path);ap.add_argument('--collection',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--exclude-index',type=Path,help='Verified teacher index; select later roots of previously sampled worlds')
    ap.add_argument('--minimum-frame-gap',type=int,default=30)
    ap.add_argument('--max-roots',type=int,choices=(16,32),default=16)
    a=ap.parse_args();repo=Path(__file__).resolve().parents[1];root=a.out.resolve();assert not root.exists()
    assert bool(a.reference)!=bool(a.collection),'Choose a reused reference or a fresh source collection'
    assert a.minimum_frame_gap>0
    assert not a.exclude_index or a.collection,'Registered-root exclusion requires a source collection'
    selected_pins={};coverage=[]
    if a.collection:
        old=a.collection.resolve();seal=read(old/'complete.json');assert seal['state']=='complete' and seal['collection_only'] and not seal['training_performed']
        original=read(old/'train-plan.json');names={t['seeds'][0]:t['episode']['map'] for t in original['tasks']}
        request=read(old/'train-request.json');available=[]
        parent_sha=sha(Path(original['model_path']));registered=[]
        if a.exclude_index:
            index_path=a.exclude_index.resolve();registered=load_index(repo,index_path)
            selected_pins[str(index_path)]=sha(index_path)
            for corpus in read(index_path)['corpora']:
                manifest=(repo/corpus['manifest']).resolve();selected_pins[str(manifest)]=sha(manifest)
                for entry in read(manifest)['entries']:
                    path=Path(entry['path']).resolve();selected_pins[str(path)]=sha(path)
        for c in request['cases']:
            directory=Path(c['data']);meta=read(directory/'report.json');assert meta['numerical_verification']=='cuda_verified_v1' and not meta.get('diagnostic_only')
            assert sha(directory/'rollout.jsonl')==meta['rollout_sha256']
            frames=[]
            for line in (directory/'rollout.jsonl').open(encoding='utf-8-sig'):
                r=json.loads(line)
                if r['frame']>=110 and r['sample'].get('wide_aim'):
                    frames.append(r['frame'])
            used=[r['frame'] for r in registered if r['seed']==c['seed'] and r['source_model_sha256']==parent_sha]
            frame=choose_root_frame(frames,used,a.minimum_frame_gap)
            coverage.append(dict(seed=c['seed'],map=names[c['seed']],eligible=frame is not None,eligible_frames=len(frames),registered_frames=sorted(used),selected_frame=frame))
            if frame is not None:available.append(dict(seed=c['seed'],frame=frame,map=names[c['seed']],source_data=str(directory),rollout_sha256=meta['rollout_sha256']))
            for file in ('report.json','rollout.jsonl','cuda-verification.json'):selected_pins[str(directory/file)]=sha(directory/file)
        selected=sorted(available,key=lambda e:hashlib.sha256(('coupled-teacher32:'+e['map']).encode()).hexdigest())[:a.max_roots]
        assert len(selected)>=8,'Insufficient outcome-blind eligible roots'
        selected_pins[str(old/'train-plan.json')]=sha(old/'train-plan.json');selected_pins[str(old/'complete.json')]=sha(old/'complete.json')
    else:
        old=a.reference.resolve();assert read(old/'complete.json')['state']=='complete'
        selected=sorted(read(old/'protocol.json')['selected'],key=lambda e:hashlib.sha256(('teacher-pilot:'+str(e['seed'])).encode()).hexdigest())[:2]
        original=read(old/'after-plan.json')
        selected_pins={str(old/'protocol.json'):sha(old/'protocol.json'),str(old/'after-plan.json'):sha(old/'after-plan.json')}
    source=Path(original['model_path']);source_model=read(source)
    assert not any(k.startswith('diagnostic_') for k in source_model)
    root.mkdir(parents=True);(root/'models').mkdir()
    os.environ['GOCACHE']=str(repo/'workspace/build/gocache');os.environ['Q2COOPBOT_DIAGNOSTIC_ENGINE_ROOT']=str(repo/'workspace/artifacts/combat-objective-engine-v2-20261025')
    pins={str(source):sha(source),**selected_pins}
    def run(label,cmd):
        save(root/'progress.json',dict(stage=label))
        with (root/(label+'.log')).open('w',encoding='utf-8') as log:p=subprocess.run(list(map(str,cmd)),cwd=repo,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert p.returncode==0,str(root/(label+'.log'))
    try:
        planner=root/'q2episode.exe';exporter=root/'q2ppo-data.exe'
        run('build-planner',['go','build','-o',planner,'./cmd/q2episode']);run('build-exporter',['go','build','-o',exporter,'./cmd/q2ppo-data'])
        plans=[];assignments=[]
        for ctx in selected:
            task=next(t for t in original['tasks'] if t['seeds']==[ctx['seed']])
            for mode in ('baseline','choice0','choice1'):
                model=source
                if mode!='baseline':
                    value=copy.deepcopy(source_model);value['diagnostic_wide_gate_probe']=dict(version=PROBE_VERSION,frame=ctx['frame'],choice=int(mode[-1]))
                    model=root/'models'/f"{ctx['seed']}-{mode}.json";save(model,value)
                    value.pop('diagnostic_wide_gate_probe');assert value==source_model
                plan=copy.deepcopy(original);plan['tasks']=[task];label=f"{ctx['seed']}-{mode}"
                plan.update(model_path=str(model),model_sha256=sha(model),output_root=str(root/('captures-'+label)))
                path=root/(label+'-plan.json');save(path,plan);run('verify-'+label,[planner,'--verify-plan',path,'--root',repo]);plans.append(path)
                assignments.append(dict(plan_index=len(plans)-1,seed=ctx['seed'],frame=ctx['frame'],mode=mode,model=str(model),model_sha256=sha(model)))
                pins.update({str(path):sha(path),str(model):sha(model)})
        for name in ('run_combat_wide_gate_teacher_pilot.py','combat_teacher_root_selection.py','combat_teacher_native_receipt.py','combat_teacher_movement_receipt.py','combat_teacher_ownership.py','combat_native_teacher_registry.py','combat_wide_aim.py','combat_joint_aim_policy.py','finalize_combat_cuda_rollout.py'):
            path=repo/'scripts'/name;pins[str(path)]=sha(path)
        save(root/'protocol.json',dict(state='frozen',version='combat_coupled_wide_teacher_pilot_v1',assignments=assignments,inputs=pins,source_collection=str(old),selected=selected,coverage=coverage,
            diagnostic_only=True,ordinary_training_eligible=False,training_performed=False,promotion=False,max_instances=16,timescale=2,native_cases=3*len(selected),cuda_cases=3*len(selected),
            selection='Collection: earliest eligible wide event>=110 per map; with exclude-index, first event at least minimum_frame_gap after last registered same-source root. Fixed SHA map order with declared max_roots cap, full collection coverage retained. Reused pilot: first2SHA seeds from5roots. No outcome/damage filter; all matched native branches required.',
            max_roots=a.max_roots,
            exclude_index=str(a.exclude_index.resolve()) if a.exclude_index else None,minimum_frame_gap=a.minimum_frame_gap,
            terminal_receipt_version='combat_teacher_admitted_terminal_v2',ownership_version='combat_teacher_active_combat_ownership_v2',
            acceptance='Exact observed/command prefix and root observation. Clamp0/1 choices verified by CUDA and native receipts; matching clamp must preserve full admitted first-life trajectory and all terminal outcomes. Monitor aim_by_class and movement are independently reconstructed from the complete trace, then only preparation frames before first server-dispatched dataset step are subtracted; admitted counters remain exact. Original full-monitor discrepancy retained. Active rules control invalidates labels;only a native-verified quiet terminal goal-stop receipt with no movement/attack/impulse or damage is passive. Current clamps have diagnostic densities, never ordinary PPO data.'))
        run('collect',[shutil.which('pwsh'),'-NoProfile','-Command',pool_command(repo,plans,16,54400,root/'pool')])
        pool=read(root/'pool/report.json');assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==len(pool['jobs'])==3*len(selected)
        tasks=[];pairs={};exports=[]
        for i,j in enumerate(sorted(pool['jobs'],key=lambda j:j['plan_index'])):
            assert not j['error'];assignment=assignments[j['plan_index']];cap=read(Path(j['root'])/'report.json')['results'][0]
            steps=[json.loads(l) for l in (Path(cap['root'])/'dataset/steps.jsonl').open(encoding='utf-8-sig')]
            pairs.setdefault(j['seed'],{})[assignment['mode']]=(cap,first_life(cap,steps))
            native=root/f'export-{i:02d}-native';cuda=root/f'export-{i:02d}-cuda'
            run('export-'+str(i),[exporter,'--batch',j['root'],'--model',assignment['model'],'--cuda-only','--out',native])
            tasks.append(dict(model=assignment['model'],model_sha256=assignment['model_sha256'],data=str(native),out=str(cuda),native_report_sha256=sha(native/'report.json')))
            exports.append(dict(**assignment,data=str(cuda)))
        save(root/'cuda-request.json',dict(version='combat_cuda_finalize_batch_v1',compact_finalized_streams=True,tasks=tasks))
        run('cuda',[sys.executable,repo/'scripts/finalize_combat_cuda_batch.py','--request',root/'cuda-request.json','--receipt-root',root/'cuda-batch'])
        assert read(root/'cuda-batch/report.json')['state']=='complete'
        results=[]
        for seed,modes in pairs.items():
            frame=next(c['frame'] for c in selected if c['seed']==seed);bc,bb=modes['baseline']
            projected=lambda rows:[dict(observation=s['observation'],sent_command=s['sent_command'],owner=s['owner']) for s in rows]
            baseline_root=next(s for s in bb if s['observation']['identity']['frame']==frame);choice=baseline_root['sample']['wide_aim']['choice']
            outcomes={};terminal_scope={}
            for mode,(cap,rows) in modes.items():
                assert cap['generated_fixture']==bc['generated_fixture']
                prefix=lambda rr:[s for s in projected(rr) if s['observation']['identity']['frame']<frame]
                assert first_diff(prefix(bb),prefix(rows)) is None,dict(seed=seed,mode=mode,error='prefix')
                current=next(s for s in rows if s['observation']['identity']['frame']==frame)
                assert first_diff(current['observation'],baseline_root['observation']) is None
                if mode!='baseline':
                    assert current['sample']['wide_aim']['choice']==int(mode[-1])
                    if int(mode[-1])==choice:
                        assert first_diff(projected(bb),projected(rows)) is None,'Matching clamp altered trajectory'
                        first_frame=bb[0]['observation']['identity']['frame']
                        assert rows[0]['observation']['identity']['frame']==first_frame
                        left,left_proof=admitted_terminal_receipt(bc,first_frame)
                        right,right_proof=admitted_terminal_receipt(cap,first_frame)
                        assert first_diff(left,right) is None,'Matching clamp altered admitted terminal'
                        terminal_scope=dict(baseline=left_proof,matching_clamp=right_proof,
                            full_monitor_terminal_difference=first_diff(terminal_receipt(bc),terminal_receipt(cap)))
                d=cap['first_life']['damage'];end=cap['goal_stop']['kill_frame'] if cap['goal_stop'] else cap['first_life']['end_frame']
                ownership=rules_ownership(rows,cap['goal_stop'],cap['first_life']['end_frame'])
                outcomes[mode]=dict(goal=bool(cap['goal_stop']),death=bool(cap['death_stop'] or cap['first_life']['minimum_health']<=0),kills=sum(v['monster_kills'] for v in d['by_mod']),dealt=sum(v['monster_health_damage'] for v in d['by_mod']),received=d['received_health_damage'],rules_combat_rows=ownership['rules_combat_rows'],passive_goal_stop_rows=len(ownership['passive_goal_stop_frames']),rules_ownership_audit=ownership,root_choice=current['sample']['wide_aim']['choice'],root_attack=current['sample']['attack'],root_sent=current['sent_command'])
            results.append(dict(seed=seed,map=bc['first_life']['map'],frame=frame,prefix_exact=True,current_observation_exact=True,matching_clamp_first_life_exact=True,monitor_aim_reconciliation=terminal_scope,outcomes=outcomes))
        for p,h in pins.items():assert sha(p)==h
        save(root/'exports.json',dict(entries=exports,diagnostic_only=True,ordinary_training_eligible=False,scope='All branches including baseline are diagnostic references; reused TRAIN, no fitting'))
        save(root/'results.json',dict(state='complete',cases=results,diagnostic_only=True,training_performed=False,promotion=False))
        save(root/'complete.json',dict(state='complete',native_cases=3*len(selected),cuda_cases=3*len(selected),results_sha256=sha(root/'results.json'),exports_sha256=sha(root/'exports.json'),diagnostic_only=True,training_performed=False,promotion=False))
        save(root/'progress.json',dict(stage='complete'));print(root/'results.json')
    except BaseException as e:save(root/'progress.json',dict(stage='failed',error=str(e)));raise

if __name__=='__main__':main()
