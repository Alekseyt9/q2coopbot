"""Test preselected shooting-preserving moves on two unseen future RNG streams."""
import argparse
import json
import math
from pathlib import Path
import statistics
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();root=a.collection.resolve();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or expected==h,str(path);pins[key]=h;return path
    pin(__file__)
    closed=read(pin(root/'complete.json'));assert closed['version']=='combat_native_continuation_pairs_v1'
    spec=read(pin(root/'spec.json',closed['spec_sha256']));p=read(pin(spec['protocol'],spec['protocol_sha256']))
    assert p['study']=='combat_underfire_constrained_confirmation_v1' and closed['native_cases']==closed['cuda_cases']==26
    folder=Path(p['selection_out']);ss=read(pin(folder/'complete.json'))
    selection=read(pin(folder/'report.json',ss['report_sha256']));assert selection['protocol_sha256']==sha(spec['protocol'])
    for path,h in selection['inputs'].items():pin(path,h)
    choices={item['context']['context_id']:item for item in selection['units']}
    assert len(choices)==4 and p['continuation_seeds']==[2026102404,2026102405] and p['horizon']==10

    def measure(folder,frame):
        meta=read(pin(folder/'cuda/report.json'));assert meta['numerical_verification']=='cuda_verified_v1' and meta['observation_timing_contract']=='frozen_world_v1'
        sp=next((folder/'native').glob('replay-*/steps.jsonl'))
        steps=list(map(json.loads,pin(sp,meta['source_sha256'][str(sp.resolve())]).open(encoding='utf-8-sig')))
        ep=sp.with_name('server_outcomes.jsonl');events=list(map(json.loads,pin(ep,meta['source_sha256'][str(ep.resolve())]).open(encoding='utf-8-sig')))
        effects={e['step']:e for e in events};assert len(effects)==len(events)
        active={s['observation']['identity']['frame']:s for s in steps if s['owner']=='provider' and s.get('native_step') and s['observation']['identity']['life']==1}
        chunks=[];unknown=[];fields=('received_health_damage','monster_health_damage')
        for f in range(frame,frame+p['horizon']):
            s=active.get(f)
            if s is None:unknown.append(dict(frame=f,reason='missing_provider_native_window'));continue
            dispatch=s.get('server_execution') or {};ns=s['native_step']
            if not(dispatch.get('matched') and dispatch.get('window_exclusive') and dispatch.get('recovery_commands')==0 and ns['begin_frame']==f and ns['end_frame']==f+1):
                unknown.append(dict(frame=f,reason='unaccepted_native_window'));continue
            e=effects.get(s['index'])
            if e is None or not e['available']:unknown.append(dict(frame=f,reason='unavailable_native_effect'));continue
            assert all(type(e[k]) in (int,float) and math.isfinite(e[k]) for k in fields);chunks.append(e)
        endpoint=any(o and o['identity']['life']==1 and o['identity']['frame']==frame+p['horizon'] for s in steps for o in (s.get('observation'),s.get('next_observation')))
        if not endpoint:unknown.append(dict(frame=frame+p['horizon'],reason='missing_same_life_endpoint'))
        totals={k:sum(e[k] for e in chunks) for k in fields}
        return dict(known=not unknown,effects=totals if not unknown else None,observed_totals=totals,unknown=unknown)

    units=[];groups={cid:[] for cid in choices}
    for unit in spec['contexts']:
        c=unit['context'];cid=c['context_id'];choice=choices[cid];assert choice['context']==c and c['role']=='auxiliary_train'
        ps=read(pin(root/'proofs'/unit['unit_id']/'complete.json'))
        proof=read(pin(root/'proofs'/unit['unit_id']/'report.json',ps['report_sha256']))
        for path,h in proof['inputs'].items():pin(path,h)
        assert proof['context']==c and proof['fork']['seed'] in p['continuation_seeds']
        labels={v['label']:v for v in unit['variants']};results={v['label']:v for v in proof['results']}
        assert set(results)=={'control',choice['selected']}
        measurements={}
        for label in results:
            assert results[label]['movement_window']['duration_frames']==5
            measurements[label]=measure(root/f"case-{labels[label]['plan_index']:02d}",c['frame'])
        known=proof['gate_passed'] and all(m['known'] for m in measurements.values())
        row=dict(unit_id=unit['unit_id'],context_id=cid,map=c['instance']['map'],stream_seed=proof['fork']['seed'],
            selected=choice['selected'],strict_gate_passed=proof['gate_passed'],known=known,measurements=measurements)
        if known:
            control,current=(measurements[label]['effects'] for label in ('control',choice['selected']))
            cost=lambda e:e['received_health_damage']-.25*e['monster_health_damage']
            row.update(cost_gain=cost(control)-cost(current),incoming_reduction=control['received_health_damage']-current['received_health_damage'],
                outgoing_loss=control['monster_health_damage']-current['monster_health_damage'])
        groups[cid].append(row);units.append(row)
    worlds=[]
    for cid,series in groups.items():
        assert len(series)==2 and {r['stream_seed'] for r in series}==set(p['continuation_seeds'])
        known=all(r['known'] for r in series);row=dict(context=choices[cid]['context'],selected=choices[cid]['selected'],known=known)
        if known:row.update({field:statistics.mean(r[field] for r in series) for field in ('cost_gain','incoming_reduction','outgoing_loss')})
        worlds.append(row)
    good=[r for r in worlds if r['known']];technical=all(r['strict_gate_passed'] for r in units)
    means={field:statistics.mean(r[field] for r in good) if good else None for field in ('cost_gain','incoming_reduction','outgoing_loss')}
    useful=sum(r['cost_gain']>=2 for r in good);signal=p['signal']
    supported=bool(technical and len(good)>=signal['minimum_known_worlds'] and useful>=signal['minimum_worlds_mean_gain_at_least_2hp'] and
        means['cost_gain']>=signal['minimum_mean_cost_gain_hp'] and means['outgoing_loss']<=signal['maximum_mean_outgoing_loss_hp'])
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True);save(a.out/'units.json',units);save(a.out/'worlds.json',worlds)
    report=dict(state='complete',version='combat_underfire_constrained_confirmation_report_v1',gate_passed=technical,
        signal_supported=supported,summary=dict(known_worlds=len(good),worlds_gain_at_least_2hp=useful,means=means),
        worlds=worlds,inputs=pins,units_sha256=sha(a.out/'units.json'),worlds_sha256=sha(a.out/'worlds.json'),
        original_pilot_failure_retained=True,training_performed=False,fit_admission=False,runtime_action_admitted=False,
        ordinary_training_eligible=False,promotion=False,
        scope='Conditional replication on two unseen policy streams of four TRAIN worlds, choices frozen from old first2streams. No candidate selection on new outcomes, world generalization or learned-policy superiority claim.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(technical_gate=technical,signal_supported=supported,summary=report['summary'],worlds=worlds))


if __name__=='__main__':main()
