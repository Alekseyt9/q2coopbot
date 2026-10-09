"""Freeze equal fresh own-policy budgets for FireBC and repaired CUDA priors."""
import argparse, json, pathlib, shutil
from process_combat_architecture_pool import read, sha, save, run


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evaluation',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    ap.add_argument('--seed-offset',type=int,default=64)
    a=ap.parse_args();repo=pathlib.Path(__file__).resolve().parents[1]
    evaluation=a.evaluation.resolve();out=a.out.resolve()
    protocol=read(evaluation/'protocol.json');pool=read(evaluation/'pool/report.json')
    quality=read(evaluation/'quality-report.json')
    assert pool['state']=='complete' and pool['source_unchanged']
    assert len(pool['jobs'])==400 and all(not j['error'] for j in pool['jobs'])
    assert quality and a.seed_offset>=64 and not out.exists()
    entries=[protocol['evaluations'][i] for i in (0,1,3)]
    out.mkdir();shutil.copy2(evaluation/'q2episode.exe',out/'q2episode.exe')
    bindings=[];plans=[];canonical=None
    names=['firebc','attention128-seed-20261007','attention128-seed-20261008']
    for name,entry in zip(names,entries):
        assert sha(entry['plan'])==entry['plan_sha256']
        template=read(entry['plan']);source=pathlib.Path(entry['source_weights'])
        assert sha(source)==entry['source_weights_sha256']
        assert sha(template['registry_path'])==template['registry_sha256']
        folder=out/name;folder.mkdir();model=folder/'weights.json'
        value=read(source);assert value['kind']=='combat_ppo_v1' and not value['deterministic']
        shutil.copy2(source,model);path=folder/'plan.json'
        run([out/'q2episode.exe','--registry',template['registry_path'],'--root',repo,
             '--episodes',','.join(protocol['families']),'--split','train','--mode','learned',
             '--count',4,'--seed-offset',a.seed_offset,'--out',path,
             '--artifacts',folder/'capture','--model',model],folder/'compile.log')
        plan=read(path)
        conditions=[dict(episode=t['episode'],seeds=t['seeds'],instances=t.get('instances',[])) for t in plan['tasks']]
        if canonical is None:canonical=conditions
        assert conditions==canonical and sum(len(t['seeds']) for t in plan['tasks'])==80
        assert all(t['split']=='train' and t['modes']==['learned'] for t in plan['tasks'])
        validation_seeds={s for t in template['tasks'] for s in t['seeds']}
        assert not validation_seeds.intersection(s for t in plan['tasks'] for s in t['seeds'])
        bindings.append(dict(id=name,architecture=dict(id='attention64' if name=='firebc' else 'attention128',
            architecture='attention',width=64 if name=='firebc' else 128),
            initialization_seed=20261008 if name.endswith('20261008') else 20261007,
            model=str(model),plan=str(path),capture_root=plan['output_root'],
            parent_weights=str(source),parent_weights_sha256=sha(source)))
        plans.append(str(path))
    save(out/'models.json',bindings);save(out/'plans.json',plans)
    save(out/'protocol.json',dict(version='combat_distillation_own_policy_training_v1',
        evaluation_protocol_sha256=sha(evaluation/'protocol.json'),evaluation_pool_sha256=sha(evaluation/'pool/report.json'),
        seed_offset=a.seed_offset,split='train',episodes_per_model=80,total_episodes=240,slots=16,timescale=2,
        scope='Equal additional own-policy episode budget; inherited experience differs. Fresh optimizers, no validation gradients; retention weights zero. No architecture superiority claim.'))
    print(json.dumps(dict(root=str(out),episodes=240,conditions_identical=True)))


if __name__=='__main__':main()
