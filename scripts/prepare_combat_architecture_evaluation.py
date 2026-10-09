"""Freeze paired native validation plans after an own-policy CUDA update batch."""
import argparse, json, pathlib, shutil
from process_combat_architecture_pool import read, sha, save, run


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--processing-root',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    ap.add_argument('--compiler',type=pathlib.Path,required=True)
    ap.add_argument('--teacher',type=pathlib.Path)
    ap.add_argument('--before-processing-root',type=pathlib.Path)
    ap.add_argument('--seed-offset',type=int,default=24)
    ap.add_argument('--count',type=int,default=4)
    args=ap.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[1]
    processing=args.processing_root.resolve(); out=args.out.resolve()
    report=read(processing/'report.json')
    assert report['state']=='complete' and report['training']
    assert args.count>=4 and args.count%4==0 and args.seed_offset>=0
    assert not out.exists(),'Fresh evaluation root required'
    capture=pathlib.Path(report['protocol']['capture_root'])
    assert sha(capture/'pool/report.json')==report['protocol']['pool_sha256']
    assert sha(capture/'models.json')==report['protocol']['models_sha256']
    bindings=read(capture/'models.json')
    before_updates={}
    if args.before_processing_root:
        before_report=read(args.before_processing_root/'report.json')
        assert before_report['state']=='complete'
        before_updates={u['model']:u for u in before_report['training']}
    assert len(bindings)==len(report['training']) and len({b['id'] for b in bindings})==len(bindings)
    trainplans=[read(x['plan']) for x in bindings]
    assert sha(trainplans[0]['registry_path'])==trainplans[0]['registry_sha256'],'Registry changed since training'
    assert all(p['registry_sha256']==trainplans[0]['registry_sha256'] for p in trainplans)
    assert len({u['allocated_episodes'] for u in report['training']})==1,'Unequal allocated training budget'
    families=[t['episode']['id'] for t in trainplans[0]['tasks']]
    assert all([t['episode']['id'] for t in p['tasks']]==families for p in trainplans)
    out.mkdir();shutil.copy2(args.compiler,out/'q2episode.exe')
    shutil.copy2(pathlib.Path(__file__),out/'prepare.py')
    evaluations=[];plans=[];canonical=None
    entries=[]
    for binding,update in zip(bindings,report['training']):
        assert binding['id']==update['model'] and update['device']=='cuda'
        assert sha(update['weights'])==update['weights_sha256']
        seal=read(pathlib.Path(update['weights']).parent/'complete.json')
        for filename,field in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
            assert sha(pathlib.Path(update['weights']).parent/filename)==seal[field]
        before=binding['model']
        if args.before_processing_root:
            original=before_updates[binding['id']];before=original['weights']
            original_root=pathlib.Path(before).parent;original_seal=read(original_root/'complete.json')
            for filename,field in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
                assert sha(original_root/filename)==original_seal[field]
            assert sha(before)==original['weights_sha256']
        entries.extend([(binding['id'],'before',before,'learned'),
                        (binding['id'],'after',update['weights'],'learned')])
    if args.teacher:
        entries.append(('firebc','baseline',str(args.teacher.resolve()),'learned'))
    entries.append(('rules','baseline',None,'rules'))
    for model,label,source,mode in entries:
        folder=out/(model+'-'+label);folder.mkdir();weights=None
        source_hash=None
        if source:
            source_hash=sha(source);weights=folder/'weights.json';value=read(source)
            assert value['kind']=='combat_ppo_v1'
            value['deterministic']=True
            save(weights,value)
        planpath=folder/'plan.json';capture_root=folder/'capture'
        command=[out/'q2episode.exe','--registry',trainplans[0]['registry_path'],'--root',repo,
                 '--episodes',','.join(families),'--split','validation','--mode',mode,
                 '--count',args.count,'--seed-offset',args.seed_offset,'--out',planpath,'--artifacts',capture_root]
        if weights:command.extend(['--model',weights])
        run(command,folder/'compile.log');plan=read(planpath)
        scenes=[dict(episode=t['episode']['id'],seeds=t['seeds'],instances=t.get('instances',[])) for t in plan['tasks']]
        if canonical is None:canonical=scenes
        assert scenes==canonical,'Before/after/baselines must share exact native starts and seeds'
        assert all(t['split']=='validation' for t in plan['tasks'])
        evaluations.append(dict(model=model,label=label,root=str(capture_root),plan=str(planpath),
                                plan_sha256=sha(planpath),source_weights_sha256=source_hash,
                                deterministic_weights_sha256=sha(weights) if weights else None))
        plans.append(str(planpath))
    save(out/'plans.json',plans)
    save(out/'protocol.json',dict(version='combat_architecture_paired_validation_v1',
         processing_report_sha256=sha(processing/'report.json'),evaluations=evaluations,
         before_processing_report_sha256=sha(args.before_processing_root/'report.json') if args.before_processing_root else None,
         episodes_per_model=len(families)*args.count,total_episodes=len(entries)*len(families)*args.count,
         families=families,seed_offset=args.seed_offset,split='validation',final_test_deferred=True,
         validation_reused_for_tuning=True,slots=16,timescale=2,
         scope='Deterministic before/after and rules on identical native starts, optional separate FireBC baseline. Equal additional training episodes; inherited experience can differ. Reused validation seeds; not final-test superiority.'))
    save(out/'progress.json',dict(stage='plans_prepared',evaluations=len(evaluations)))
    print(json.dumps(dict(root=str(out),plans=len(plans),episodes=len(entries)*len(families)*args.count)))


if __name__=='__main__':
    main()
