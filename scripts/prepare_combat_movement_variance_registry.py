"""Pin fresh alternate-site corpus and prove reward cloning preserves fixtures."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();repo=Path(__file__).resolve().parents[1];pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or expected==h,str(path);pins[str(path)]=h;return path
    p=read(pin(a.protocol));assert p['version']=='combat_movement_variance_confirmation_v1'
    source=Path(p['source']);closed=read(pin(source/'complete.json'));pin(source/'weights.json',p['source_weights_sha256'])
    assert closed['weights_sha256']==p['source_weights_sha256']
    registry=pin(p['registry']);pin(p['config']);pin(p['reward']);catalog=read(registry)
    names=p['episodes'];assert len(names)==len(set(names))==24
    for name in names:
        assert name+'.json' in catalog['files'];e=read(pin(registry.parent/(name+'.json')))
        assert e['status']=='runnable' and e['ppo_trainable'] and e['timescale']==2
    a.out.mkdir(parents=True);os.environ['GOCACHE']=str(repo/'workspace/build/gocache');os.environ['GOTOOLCHAIN']='auto'
    def run(label,command):
        with (a.out/(label+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(list(map(str,command)),cwd=repo,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert result.returncode==0,str(a.out/(label+'.log'))
    cloned=a.out/'registry'
    run('registry',[sys.executable,repo/'scripts/prepare_combat_objective_registry.py','--registry',registry,'--config',p['config'],'--reward',p['reward'],'--episodes',','.join(names),'--out',cloned])
    planner=a.out/'q2episode.exe';run('build',['go','build','-o',planner,'./cmd/q2episode'])
    plans=[]
    for label,index in [('canonical',registry),('cloned',cloned/'index.json')]:
        path=a.out/(label+'-plan.json');run(label,[planner,'--registry',index,'--root',repo,'--episodes',','.join(names),'--split','train','--mode','learned','--model',source/'weights.json',
            '--count',1,'--seed-offset',0,'--checked-wall-slide','--out',path,'--artifacts',a.out/(label+'-unused-captures')])
        plans.append(read(pin(path)))
    left={t['episode']['id']:t for t in plans[0]['tasks']};right={t['episode']['id']:t for t in plans[1]['tasks']}
    assert set(left)==set(right)==set(names)
    for name in names:assert left[name]['seeds']==right[name]['seeds'] and left[name]['instances']==right[name]['instances'],name
    for path,h in pins.items():assert sha(path)==h,path
    report=dict(state='complete',version='combat_movement_variance_registry_preparation_v1',registry=str((cloned/'index.json').resolve()),registry_sha256=sha(cloned/'index.json'),
        source=str(source.resolve()),episodes=names,technical_fixture_seeds_and_instances_exact=True,technical_plans_only=True,
        protocol=str(a.protocol.resolve()),protocol_sha256=sha(a.protocol),inputs=pins,training_performed=False,new_games=False,promotion=False)
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(episodes=len(names),technical_fixtures_exact=True,registry=report['registry']))


if __name__=='__main__':main()
