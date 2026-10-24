"""Clone registered conditions with one explicitly pinned training objective."""
import argparse
import copy
import pathlib
from process_combat_architecture_pool import read, save, sha


def main():
    ap=argparse.ArgumentParser()
    for name in ('registry','config','reward','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--episodes',required=True)
    args=ap.parse_args();repo=pathlib.Path(__file__).resolve().parents[1]
    receipt=args.out.with_name(args.out.name+'-complete.json')
    assert not args.out.exists() and not receipt.exists()
    registry,config,reward=[p.resolve() for p in (args.registry,args.config,args.reward)]
    assert read(config)['objective_reward_sha256']==sha(reward), 'Config/reward objective mismatch'
    relative=str(reward.relative_to(repo)).replace('\\','/')
    names=args.episodes.split(',');assert names and len(names)==len(set(names))
    index=read(registry);descriptors={}
    for file in index['files']:
        path=(registry.parent/file).resolve();value=read(path)
        assert value['id'] not in descriptors;descriptors[value['id']]=(path,value)
    assert set(names)<=set(descriptors)
    pins={str(p):sha(p) for p in (registry,config,reward)};outputs=[];files=[]
    # Validate the complete request before creating any output.
    for name in names:
        _,episode=descriptors[name]
        assert episode['status']=='runnable' and episode['ppo_trainable'] and episode['timescale']==2
        assert episode['recipe']['runner']=='combat-baseline'
    args.out.mkdir(parents=True)
    for name in names:
        path,episode=descriptors[name];pins[str(path)]=sha(path)
        updated=copy.deepcopy(episode);updated['recipe']['reward_config']=relative
        if updated.get('generator'):
            # Revision participates in fixture RNG unless explicitly pinned.
            # A reward-only clone must preserve the original effective seed.
            updated['generator']['seed_revision']=episode['generator'].get('seed_revision',episode['revision'])
        updated['revision']+=1
        # Positions, generator, split ranges, monsters and loadout are untouched.
        a,b=copy.deepcopy(episode),copy.deepcopy(updated)
        a['revision']=b['revision'];a['recipe']['reward_config']=b['recipe']['reward_config']
        if a.get('generator'):a['generator']['seed_revision']=a['generator'].get('seed_revision',episode['revision'])
        assert a==b
        filename=name+'.json';assert pathlib.Path(filename).name==filename
        destination=args.out/filename;save(destination,updated);files.append(filename)
        outputs.append(dict(path=str(destination.resolve()),sha256=sha(destination),source=str(path),source_sha256=pins[str(path)]))
    save(args.out/'index.json',dict(version=index['version'],files=files))
    for path,digest in pins.items():assert sha(path)==digest
    # The Go registry compiler rejects unrelated JSON inside the registry folder.
    save(receipt,dict(state='complete',version='combat_objective_registry_v1',
        registry_sha256=sha(args.out/'index.json'),objective_reward_sha256=sha(reward),
        config_sha256=sha(config),inputs=[dict(path=p,sha256=h) for p,h in pins.items()],outputs=outputs,
        script_sha256=sha(__file__),training_performed=False,
        scope='Reward binding/revision changed; generated descriptors explicitly pin original effective seed revision. Source physical conditions/loadout/splits and fixture RNG preserved. Collector must validate the checkpoint objective and choose fresh seeds.'))
    print(args.out/'index.json')


if __name__=='__main__':main()
