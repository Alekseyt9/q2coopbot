"""Freeze a registry training round from sealed PPO checkpoints."""
import argparse,json,pathlib,shutil
from process_combat_architecture_pool import read,sha,save,run


def main():
    ap=argparse.ArgumentParser()
    for name in ('parent','compiler','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--models',required=True);ap.add_argument('--seed-offset',type=int,required=True);a=ap.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[1];parent=a.parent.resolve();out=a.out.resolve()
    report=read(parent/'report.json');assert report['state']=='complete' and not out.exists()
    capture=pathlib.Path(report['protocol']['capture_root'])
    assert sha(capture/'models.json')==report['protocol']['models_sha256']
    assert sha(capture/'pool/report.json')==report['protocol']['pool_sha256']
    bindings={b['id']:b for b in read(capture/'models.json')};updates={u['model']:u for u in report['training']}
    names=a.models.split(',');assert len(set(names))==len(names) and a.seed_offset>=72
    out.mkdir();shutil.copy2(a.compiler,out/'q2episode.exe');plans=[];models=[];canonical=None
    for name in names:
        binding=bindings[name];update=updates[name];source=pathlib.Path(update['weights']).parent
        assert binding['architecture']['architecture']=='attention' and update['device']=='cuda'
        seal=read(source/'complete.json')
        for filename,field in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
            assert sha(source/filename)==seal[field]
        oldplan=read(binding['plan']);assert sha(oldplan['registry_path'])==oldplan['registry_sha256']
        folder=out/name;folder.mkdir();prior=folder/'parent';prior.mkdir()
        for filename in ('weights.json','checkpoint.pt','report.json','complete.json'):shutil.copy2(source/filename,prior/filename)
        value=read(prior/'weights.json');assert not value['deterministic']
        path=folder/'plan.json';weight=prior/'weights.json'
        run([out/'q2episode.exe','--registry',oldplan['registry_path'],'--root',repo,
             '--episodes',','.join(t['episode']['id'] for t in oldplan['tasks']),
             '--split','train','--mode','learned','--count',4,'--seed-offset',a.seed_offset,
             '--out',path,'--artifacts',folder/'capture','--model',weight],folder/'compile.log')
        plan=read(path);conditions=[(t['episode'],t['seeds'],t.get('instances',[])) for t in plan['tasks']]
        if canonical is None:canonical=conditions
        assert conditions==canonical and sum(len(t['seeds']) for t in plan['tasks'])==80
        assert all(t['split']=='train' and t['modes']==['learned'] for t in plan['tasks'])
        assert not {s for t in oldplan['tasks'] for s in t['seeds']}.intersection(s for t in plan['tasks'] for s in t['seeds'])
        parent_report=read(prior/'report.json')
        models.append(dict(id=name,architecture=binding['architecture'],initialization_seed=binding['initialization_seed'],
            model=str(weight),plan=str(path),capture_root=plan['output_root'],
            resume_checkpoint=str(prior/'checkpoint.pt'),resume_checkpoint_sha256=sha(prior/'checkpoint.pt'),
            resume_report=str(prior/'report.json'),resume_report_sha256=sha(prior/'report.json'),
            parent_updates_completed=parent_report['updates_completed']))
        plans.append(str(path))
    save(out/'models.json',models);save(out/'plans.json',plans)
    save(out/'protocol.json',dict(version='combat_checkpoint_training_v1',parent_processing_root=str(parent),
        parent_report_sha256=sha(parent/'report.json'),seed_offset=a.seed_offset,split='train',slots=16,timescale=2,
        episodes_per_model=80,total_episodes=80*len(models),optimizer_state='continue_sealed_checkpoint',models_sha256=sha(out/'models.json')))
    print(json.dumps(dict(root=str(out),episodes=80*len(models),seed_offset=a.seed_offset)))


if __name__=='__main__':main()
