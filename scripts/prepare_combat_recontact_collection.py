"""Geometry-only registry expansion for fresh native recontact data collection."""
import argparse
from pathlib import Path
import re
import subprocess
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];out=a.out.resolve();assert not out.exists();out.mkdir(parents=True)
    planner=repo/'workspace/build/q2episode.exe';assert planner.exists()
    selections=[];catalogs=[];failures=[];registry=out/'registry';registry.mkdir()
    for name in ('base3','city1','jail1','ware2'):
        target=out/(name+'-catalog.json')
        command=[str(planner),'--root',str(repo),'--campaign-sites','4','--campaign-maps',name,
            '--campaign-group','monster_parasite,monster_infantry','--campaign-group-loadout','weapons','--out',str(target)]
        p=subprocess.run(command,text=True,capture_output=True)
        if p.returncode:
            match=re.search(r'only (\d+) supported distinct sites',p.stderr)
            assert match and int(match[1])>=2,p.stderr
            failures.append(dict(map=name,requested=4,error=p.stderr.strip()))
            command[command.index('--campaign-sites')+1]=match[1]
            p=subprocess.run(command,text=True,capture_output=True)
        assert p.returncode==0,p.stderr
        catalog=read(target);episodes=catalog['episodes'];assert len(episodes)>=2
        chosen=episodes[-2:]
        assert chosen[0]['generator']['site']['entity_index']!=chosen[1]['generator']['site']['entity_index']
        for ep in chosen:
            ep['generator']['distributions']['train']['health']=[100]
            ep['recipe']['reward_config']='scripts/scenarios/combat-reward-blocked-ground-cost-v14.json'
            objective=read(repo/ep['recipe']['reward_config'])
            assert objective['version']=='combat_reward_v14' and objective['handoff_bootstrap']=='nonterminal_v1'
            save(registry/(ep['id']+'.json'),ep)
            selections.append(ep)
        catalogs.append(dict(path=str(target),sha256=sha(target),selected=[e['id'] for e in chosen]))
    save(registry/'index.json',dict(version=1,files=[e['id']+'.json' for e in selections]))
    cases=[]
    for i,ep in enumerate(selections):
        plan=out/f'anchor-plan-{i:02d}.json'
        p=subprocess.run([str(planner),'--root',str(repo),'--registry',str(registry/'index.json'),'--episodes',ep['id'],
            '--split','train','--mode','rules','--count','1','--seed-offset','43','--out',str(plan),
            '--artifacts',str(out/f'anchor-not-executed-{i:02d}')],text=True,capture_output=True)
        assert p.returncode==0,p.stderr
        task=read(plan)['tasks'][0];instance=task['instances'][0]
        assert instance['health']==100
        cases.append(dict(map=ep['map'],episode=ep['id'],seed=instance['engine_seed'],loadout=ep['recipe']['loadout'],
            frame=100,source_plan=str(plan),instance=instance,selection_origin='Geometry-generated anchor plan; no native battle executed here'))
    assert len(cases)==8 and len({c['seed'] for c in cases})==8
    source=repo/'workspace/artifacts/combat-weapon-only-initial-v1-20261024/control/weights.json'
    save(out/'selection.json',dict(version='combat_geometry_selected_recontact_collection_v1',source_model=str(source),
        source_weights_sha256=sha(source),cases=cases,selection='Last two supported distinct BSP sites of four requested per map; outcome-independent',
        catalogs=catalogs,generation_limits=failures,planner_sha256=sha(planner),native_execution=False,promotion=False))
    print([(c['map'],c['episode'],c['seed']) for c in cases],flush=True)


if __name__=='__main__':main()
