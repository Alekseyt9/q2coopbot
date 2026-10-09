"""Declared provider aim modes, stratified by observed target range and weapon."""
import argparse,collections,json,pathlib
from process_combat_architecture_pool import read,save,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);a=ap.parse_args()
    protocol=read(a.root/'protocol.json');proof=read(a.root/'recovery/verified-members.json');assert proof['state']=='complete' and proof['protocol_sha256']==sha(a.root/'protocol.json')
    groups={}
    for entry in protocol['evaluations']:
        plan=read(entry['plan']);model=read(pathlib.Path(plan['model_path'])) if plan.get('model_path') else {}
        if not model.get('aim_mode_head'):
            groups[entry['model']+'-'+entry['label']]=dict(status='mode_not_declared');continue
        buckets={};switches=0;adjacent=0
        for path in pathlib.Path(entry['root']).glob('case-*/s-*/report.json'):
            assert proof['members'][str(path.parent)]['report_sha256']==sha(path)
            result=read(path)['results'][0];previous=None
            with (pathlib.Path(result['root'])/'bot.jsonl').open(encoding='utf-8-sig') as f:
                for line in f:
                    c=json.loads(line).get('combat_policy') or {};o=c.get('observation');s=c.get('selection',{});candidate=s.get('candidate',{})
                    if not o or s.get('owner')!='provider' or o['identity']['life']!=1 or o['identity']['frame']<=100 or o['health']<=0:previous=None;continue
                    mode=candidate.get('aim_mode',0);assert mode in (0,1)
                    entity=candidate.get('target_entity',0);enemy=next((e for e in o['enemies'] if e['id']==entity),None)
                    if not enemy:distance='none'
                    elif enemy['distance']<=128:distance='near'
                    elif enemy['distance']<=512:distance='medium'
                    else:distance='far'
                    key=o['weapon']+' / '+distance;count=buckets.setdefault(key,collections.Counter());count['frames']+=1;count['fine' if mode else 'coarse']+=1
                    if c['applied']['attack']:count['firing_frames']+=1
                    frame=o['identity']['frame']
                    if previous and frame==previous[0]+1:adjacent+=1;switches+=int(mode!=previous[1])
                    previous=(frame,mode)
        groups[entry['model']+'-'+entry['label']]=dict(status='declared',buckets=buckets,mode_switches=switches,adjacent_frames=adjacent)
    save(a.root/'aim-modes.json',dict(protocol_sha256=sha(a.root/'protocol.json'),groups=groups,scope='Declared candidate aim mode on current first-life provider frames; range of explicit observed target. Counts not bullet counts or mode correctness; guards may alter actual commands. Legacy modes remain unknown.'))

if __name__=='__main__':main()
