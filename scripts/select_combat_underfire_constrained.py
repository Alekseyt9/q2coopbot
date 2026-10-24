"""Freeze a shooting-preserving action using discovery training streams only."""
import argparse
from pathlib import Path
import statistics
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or expected==h,str(path);pins[str(path)]=h;return path
    p=read(pin(a.protocol));assert p['study']=='combat_underfire_constrained_confirmation_v1'
    root=Path(p['parent_pilot']);closed=read(pin(root/'dataset/complete.json'))
    report=read(pin(root/'dataset/report.json',closed['report_sha256']))
    assert report['gate_passed'] and not report['pilot_signal_supported']
    records=read(pin(root/'dataset/records.json',report['artifacts']['records']))
    cs=read(pin(root/'complete.json'));spec=read(pin(root/'spec.json',cs['spec_sha256']))
    contexts={u['context']['context_id']:u['context'] for u in spec['contexts']}
    assert len(contexts)==4 and p['selection_streams']==[2026102401,2026102402]
    assert not set(p['continuation_seeds']).intersection({r['stream_seed'] for r in records})
    chosen=[]
    for cid,context in contexts.items():
        assert context['split']=='train' and context['role']=='auxiliary_train'
        measurements={}
        for label in p['movement_labels']:
            series=[r for r in records if r['context_id']==cid and r['label']==label and r['stream_seed'] in p['selection_streams']]
            assert len(series)==2 and {r['stream_seed'] for r in series}==set(p['selection_streams'])
            effects=[r['measurements'][str(p['horizon'])] for r in series];assert all(e['known'] for e in effects)
            measurements[label]={field:statistics.mean(e['effects'][field] for e in effects) for field in ('received_health_damage','monster_health_damage')}
        control=measurements['control'];cost=lambda e:e['received_health_damage']-.25*e['monster_health_damage']
        eligible=[label for label in p['movement_labels'] if control['monster_health_damage']-measurements[label]['monster_health_damage']<=p['maximum_selection_mean_outgoing_loss_hp']]
        selected=min(eligible,key=lambda label:cost(measurements[label]))
        chosen.append(dict(context=context,selected=selected,eligible=eligible,selection_mean_effects=measurements,
            selection_cost_gain=cost(control)-cost(measurements[selected]),
            selection_outgoing_loss=control['monster_health_damage']-measurements[selected]['monster_health_damage']))
    count=sum(2*(2+int(item['selected']!='control'))+1 for item in chosen)
    assert count==p['native_cases']==p['cuda_cases']==26
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True);report=dict(state='complete',version='combat_underfire_constrained_selection_v1',units=chosen,
        protocol_sha256=sha(a.protocol),inputs=pins,training_performed=False,neural_execution=False,
        old_held_stream_used_for_selection=False,new_confirmation_streams_used=False,promotion=False)
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(cases=count,selected=[dict(map=u['context']['instance']['map'],selected=u['selected'],eligible=u['eligible']) for u in chosen]))


if __name__=='__main__':main()
