"""New harness cohort, preserving all frozen paired conditions and weights."""
import argparse, copy, hashlib, json, pathlib, shutil

def read(path): return json.loads(pathlib.Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def save(path,value): pathlib.Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--from-evaluation',type=pathlib.Path,required=True)
    ap.add_argument('--processing-root',type=pathlib.Path,required=True); ap.add_argument('--out',type=pathlib.Path,required=True)
    args=ap.parse_args(); source=args.from_evaluation.resolve(); out=args.out.resolve(); processing=args.processing_root.resolve()
    repo=pathlib.Path(__file__).resolve().parents[1]; original=read(source/'protocol.json')
    assert not out.exists(),'Fresh cohort required'
    assert sha(processing/'report.json')==original['processing_report_sha256']
    updates=read(processing/'training-updates.json')
    assert len(updates)==8 and all(x['device']=='cuda' for x in updates)
    for update in updates:
        folder=pathlib.Path(update['weights']).parent; seal=read(folder/'complete.json')
        for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
            assert sha(folder/name)==seal[key],f'Completed CUDA artifact changed: {folder/name}'
    validated=[]; scenes=None
    for entry in original['evaluations']:
        assert sha(entry['plan'])==entry['plan_sha256'],'Frozen plan changed'
        plan=read(entry['plan']); assert sha(plan['registry_path'])==plan['registry_sha256']
        conditions=[dict(episode=t['episode'],episode_sha256=t['episode_sha256'],split=t['split'],seeds=t['seeds'],instances=t.get('instances',[])) for t in plan['tasks']]
        if scenes is None: scenes=conditions
        assert conditions==scenes,'Pair conditions differ'
        if plan.get('model_path'): assert sha(plan['model_path'])==entry['deterministic_weights_sha256']==plan['model_sha256']
        for task in plan['tasks']:
            reward=task['episode']['recipe'].get('reward_config')
            if reward: assert sha(repo/reward)==task['reward_sha256'],'Reward changed'
        validated.append((entry,plan))
    out.mkdir(); shutil.copy2(source/'q2episode.exe',out/'q2episode.exe'); evaluations=[]; plans=[]
    for entry,oldplan in validated:
        folder=out/(entry['model']+'-'+entry['label']); folder.mkdir(); plan=copy.deepcopy(oldplan)
        plan['output_root']=str(folder/'capture')
        if plan.get('model_path'):
            shutil.copy2(plan['model_path'],folder/'weights.json'); plan['model_path']=str(folder/'weights.json')
            assert sha(plan['model_path'])==entry['deterministic_weights_sha256']
        for task in plan['tasks']: task['runner_sha256']=sha(task['runner_path'])
        path=folder/'plan.json'; save(path,plan); plans.append(str(path))
        newentry=dict(entry,root=plan['output_root'],plan=str(path),plan_sha256=sha(path)); evaluations.append(newentry)
    save(out/'plans.json',plans)
    protocol=dict(original,evaluations=evaluations,predecessor_root=str(source),predecessor_protocol_sha256=sha(source/'protocol.json'),
                  restart_reason='Storage hooks and shared AAS changed harness fingerprint. Prior partial cohort is retained and excluded.',
                  exact_previous_conditions_preserved=True,old_captures_reused=0)
    save(out/'protocol.json',protocol); save(out/'progress.json',{'stage':'plans_prepared','evaluations':len(evaluations)})
    print(json.dumps({'root':str(out),'episodes':protocol['total_episodes'],'same_weights_and_conditions':True}))

if __name__=='__main__': main()
