"""Freeze task-success-first actions from terminal-known discovery streams."""
import argparse
from pathlib import Path
import statistics
from evaluate_registered_combat_models import read,save,sha


def utility(r):return 4*float(r['goal'] and not r['death'])+.25*r['kills']-r['received']/100.-float(r['death'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or h==expected,str(path);pins[str(path)]=h;return path
    pin(__file__);p=read(pin(a.protocol));assert p['study']=='combat_underfire_full_task_replication_v1'
    root=Path(p['parent_pilot']);closed=read(pin(root/'complete.json'))
    spec=read(pin(root/'spec.json',closed['spec_sha256']));parent=read(pin(spec['protocol'],spec['protocol_sha256']))
    assert parent['continuation_seeds']==[2026102401,2026102402,2026102403] and p['selection_streams']==parent['continuation_seeds']
    groups={}
    for unit in spec['contexts']:
        c=unit['context'];ps=read(pin(root/'proofs'/unit['unit_id']/'complete.json'))
        proof=read(pin(root/'proofs'/unit['unit_id']/'report.json',ps['report_sha256']))
        if proof['fork']['seed'] not in p['selection_streams']:continue
        assert proof['gate_passed'] and c['role']=='auxiliary_train'
        group=groups.setdefault(c['context_id'],dict(context=c,streams={}))
        group['streams'][proof['fork']['seed']]={r['label']:r for r in proof['results']}
    units=[]
    for group in groups.values():
        streams=group['streams'];assert set(streams)==set(p['selection_streams']);scores={};excluded=[]
        for label in p['movement_labels']:
            rows=[streams[seed][label] for seed in p['selection_streams']]
            if not all(r['admitted'] and (r['goal'] or r['death']) for r in rows):
                excluded.append(dict(label=label,reason='Not strict terminal-known in all discovery streams'));continue
            scores[label]=dict(alive_goal=statistics.mean(float(r['goal'] and not r['death']) for r in rows),
                kills=statistics.mean(r['kills'] for r in rows),utility=statistics.mean(utility(r) for r in rows))
        assert 'control' in scores
        chosen=max(scores,key=lambda label:(scores[label]['alive_goal'],scores[label]['kills'],scores[label]['utility'],-p['movement_labels'].index(label)))
        units.append(dict(context=group['context'],selected=chosen,scores=scores,excluded=excluded))
    assert len(units)==4
    count=sum(p['continuation_replicates']*(2+int(u['selected']!='control'))+1 for u in units)
    assert count==p['native_cases']==p['cuda_cases']==37
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True);report=dict(state='complete',version='combat_underfire_full_task_selection_v1',units=units,
        protocol_sha256=sha(a.protocol),inputs=pins,training_performed=False,neural_execution=False,promotion=False,
        scope='Action selection uses only all three balanced terminal-known discovery streams. Full task lexicographic priority; no candidate search on new streams.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(cases=count,selected=[dict(map=u['context']['instance']['map'],selected=u['selected']) for u in units]))


if __name__=='__main__':main()
