"""Select fresh diagnostic worlds without consulting outcomes or preferences."""
import hashlib


def select_unseen_worlds(records,known,count):
    seen={(r['map'],r['seed']) for r in known};selected=[]
    ranked=sorted(records,key=lambda r:hashlib.sha256(('wide-future-stability-v1:'+r['id']).encode()).hexdigest())
    for record in ranked:
        world=(record['map'],record['seed'])
        if world in seen:continue
        selected.append(record);seen.add(world)
        if len(selected)==count:break
    assert len(selected)==count
    return selected
