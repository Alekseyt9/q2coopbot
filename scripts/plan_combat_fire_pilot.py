"""Select registered scenes by initial inventory before any gameplay outcome."""
import argparse
import copy
from pathlib import Path
from process_combat_architecture_pool import read,save,sha


def select(candidate,total,grenades,captures):
    assert total>=grenades>0 and total<=53
    pool=[]
    for task in candidate['tasks']:
        for instance in task['instances']:
            pool.append((task,instance))
    chosen=[];used=set();maps={}
    def take(predicate):
        available=[(t,i) for t,i in pool if t['episode']['id'] not in used and maps.get(t['episode']['map'],0)<2 and predicate(i)]
        assert available,'Not enough registered inventory-balanced scenes'
        t,i=min(available,key=lambda pair:(maps.get(pair[0]['episode']['map'],0),pair[1]['engine_seed']))
        chosen.append((t,i));used.add(t['episode']['id']);maps[t['episode']['map']]=maps.get(t['episode']['map'],0)+1
    for _ in range(grenades):take(lambda i:i['inventory']['initial_weapon']=='Grenades')
    other=total-grenades
    kinds=sorted({i['inventory']['initial_weapon'] for _,i in pool if i['inventory']['initial_weapon']!='Grenades'})
    for weapon in kinds[:other]:take(lambda i,w=weapon:i['inventory']['initial_weapon']==w)
    for _ in range(other-min(len(kinds),other)):take(lambda i:i['inventory']['initial_weapon']!='Grenades')
    selected=copy.deepcopy(candidate);selected['output_root']=str(captures.resolve());selected['tasks']=[]
    for task,instance in sorted(chosen,key=lambda pair:pair[0]['episode']['id']):
        task=copy.deepcopy(task);task['instances']=[instance];task['seeds']=[instance['engine_seed']]
        selected['tasks'].append(task)
    assert len(selected['tasks'])==total and sum(t['instances'][0]['inventory']['initial_weapon']=='Grenades' for t in selected['tasks'])==grenades
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','out','captures'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--total',type=int,default=32);p.add_argument('--grenades',type=int,default=16)
    a=p.parse_args();assert not a.out.exists()
    candidate=read(a.candidate);selected=select(candidate,a.total,a.grenades,a.captures)
    save(a.out,selected)
    save(a.out.with_suffix('.selection.json'),dict(version='combat_inventory_balanced_selection_v1',
        candidate_sha256=sha(a.candidate),selected_plan_sha256=sha(a.out),total=a.total,grenades=a.grenades,
        before_outcomes=True,selection_basis='Initial weapon only, unique descriptors, at most two cases per campaign map',
        cases=[dict(episode=t['episode']['id'],map=t['episode']['map'],seed=t['seeds'][0],
            initial_weapon=t['instances'][0]['inventory']['initial_weapon'],split=t['split']) for t in selected['tasks']]))
    print('Selected',a.total,'registered cases,',a.grenades,'Grenades,',len({t['episode']['map'] for t in selected['tasks']}),'maps')


if __name__=='__main__':main()
