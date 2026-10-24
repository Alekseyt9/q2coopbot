"""Check multi-frame movement signal on a future stream excluded from selection."""
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
    seal=read(pin(root/'complete.json'));assert seal['version']=='combat_native_continuation_pairs_v1'
    spec=read(pin(root/'spec.json',seal['spec_sha256']));p=read(pin(spec['protocol'],spec['protocol_sha256']))
    assert p['study']=='combat_underfire_horizon_pilot_v1' and seal['native_cases']==seal['cuda_cases']==76
    fields=('monster_health_damage','received_health_damage','self_health_damage','teammate_health_damage','monster_kills','deaths')
    records=[];units=[];groups={}
    for unit in spec['contexts']:
        c=unit['context'];frame=c['frame'];ps=read(pin(root/'proofs'/unit['unit_id']/'complete.json'))
        proof=read(pin(root/'proofs'/unit['unit_id']/'report.json',ps['report_sha256']))
        for path,h in proof['inputs'].items():pin(path,h)
        assert proof['context']==c and proof['fork']['fork_frame']==frame+1
        stream=proof['fork']['seed'];assert stream in p['continuation_seeds']
        assert c['split']=='train' and c['role']=='auxiliary_train'
        accepted={r['label']:r for r in proof['results']};labels={}
        controls=all(proof[k] for k in ('nofork_full_exact','zero_fork_full_exact','fork_prefix_exact'))
        for variant in unit['variants']:
            label=variant['label']
            if label not in p['movement_labels']:continue
            acceptance=accepted[label]
            technical=controls and all(acceptance[k] for k in ('prefix_exact','intervention_state_exact',
                'intervention_draws_exact','categories_exact','shared_future_draws_exact'))
            assert acceptance['movement_window']['duration_frames']==5
            folder=root/f"case-{variant['plan_index']:02d}";meta=read(pin(folder/'cuda/report.json'))
            assert meta['numerical_verification']=='cuda_verified_v1' and meta['observation_timing_contract']=='frozen_world_v1'
            step_path=next((folder/'native').glob('replay-*/steps.jsonl'))
            steps=list(map(json.loads,pin(step_path,meta['source_sha256'][str(step_path.resolve())]).open(encoding='utf-8-sig')))
            ep=step_path.with_name('server_outcomes.jsonl');effect_list=list(map(json.loads,pin(ep,meta['source_sha256'][str(ep.resolve())]).open(encoding='utf-8-sig')))
            effects={e['step']:e for e in effect_list};assert len(effects)==len(effect_list)
            active={s['observation']['identity']['frame']:s for s in steps if s['owner']=='provider' and s.get('native_step') and s['observation']['identity']['life']==1}
            snapshots={}
            for s in steps:
                for key in ('observation','next_observation'):
                    o=s.get(key)
                    if o and o['identity']['life']==1:snapshots.setdefault(o['identity']['frame'],o)
            measurements={}
            for h in p['horizons_frames']:
                chunks=[];unknown=[]
                for f in range(frame,frame+h):
                    s=active.get(f)
                    if s is None:unknown.append(dict(frame=f,reason='missing_provider_native_window'));continue
                    dispatch=s.get('server_execution') or {};ns=s['native_step']
                    if not(dispatch.get('matched') and dispatch.get('window_exclusive') and dispatch.get('recovery_commands')==0 and ns['begin_frame']==f and ns['end_frame']==f+1):
                        unknown.append(dict(frame=f,reason='unaccepted_native_window'));continue
                    e=effects.get(s['index'])
                    if e is None or not e['available']:unknown.append(dict(frame=f,reason='unavailable_native_effect'));continue
                    assert all(type(e[k]) in (int,float) and math.isfinite(e[k]) for k in fields)
                    chunks.append(e)
                endpoint=snapshots.get(frame+h)
                if endpoint is None:unknown.append(dict(frame=frame+h,reason='missing_same_life_endpoint'))
                totals={k:sum(e[k] for e in chunks) for k in fields}
                measurement=dict(known=technical and not unknown,technical_admitted=technical,known_native_steps=len(chunks),unknown=unknown,observed_totals=totals)
                if measurement['known']:measurement['effects']=totals
                measurements[str(h)]=measurement
            # Prefix/root draws are fixed across streams; future RNG never affects H1.
            first=active[frame];sample=first['sample']
            root_action={k:sample.get(k,0) for k in ('sampling_seed','latent','attack','vertical','weapon','target','aim_mode','memory','aim_candidate')}
            item=dict(context_id=c['context_id'],map=c['instance']['map'],world_seed=c['seed'],frame=frame,stream_seed=stream,
                label=label,technical_admitted=technical,full_episode_admitted=acceptance['admitted'],goal=acceptance['goal'],
                death=acceptance['death'],kills=acceptance['kills'],received=acceptance['received'],measurements=measurements,
                root_action=root_action,case=str(folder.resolve()))
            records.append(item);labels[label]=item
        assert set(labels)==set(p['movement_labels'])
        cid=c['context_id'];group=groups.setdefault(cid,dict(context=c,streams={}))
        assert group['context']==c and stream not in group['streams'];group['streams'][stream]=labels
        units.append(dict(unit_id=unit['unit_id'],context_id=cid,stream_seed=stream,strict_gate_passed=proof['gate_passed'],controls_exact=controls))
    assert len(groups)==4 and len(units)==12 and len(records)==60
    for group in groups.values():
        for label in p['movement_labels']:
            series=[group['streams'][seed][label] for seed in p['continuation_seeds']]
            assert all(v['root_action']==series[0]['root_action'] for v in series)
            good=[v['measurements']['1'] for v in series if v['measurements']['1']['known']]
            assert all(v['effects']==good[0]['effects'] for v in good)
    preferences=[]
    for cid,group in groups.items():
        for horizon in (5,10):
            for label in p['movement_labels'][1:]:
                deltas=[]
                for seed in p['continuation_seeds']:
                    actions=group['streams'][seed]
                    ctrl,current=(actions[name]['measurements'][str(horizon)] for name in ('control',label))
                    if ctrl['known'] and current['known']:
                        deltas.append(dict(stream_seed=seed,**{k:current['effects'][k]-ctrl['effects'][k] for k in fields}))
                components={}
                for field in fields:
                    values=[v[field] for v in deltas]
                    components[field]=dict(mean=statistics.mean(values) if values else None,
                        sample_std=statistics.stdev(values) if len(values)>1 else None,
                        minimum=min(values) if values else None,maximum=max(values) if values else None,
                        sign_unstable=bool(values and min(values)<0<max(values)))
                preferences.append(dict(context_id=cid,map=group['context']['instance']['map'],horizon=horizon,label=label,
                    known_streams=len(deltas),complete=len(deltas)==3,deltas=deltas,components=components))
    signal=p['pilot_signal'];assert signal['horizon']==10
    held=[]
    for cid,group in groups.items():
        streams=group['streams'];h=str(signal['horizon'])
        known=all(streams[seed][label]['measurements'][h]['known'] for seed in p['continuation_seeds'] for label in signal['action_order'])
        row=dict(context_id=cid,map=group['context']['instance']['map'],known=known)
        if known:
            def damage(seed,label):return streams[seed][label]['measurements'][h]['effects']
            def cost(e):return e['received_health_damage']-.25*e['monster_health_damage']
            means={label:statistics.mean(cost(damage(seed,label)) for seed in signal['selection_streams']) for label in signal['action_order']}
            chosen=min(signal['action_order'],key=lambda label:means[label])
            control,current=(damage(signal['held_stream'],label) for label in ('control',chosen))
            row.update(chosen=chosen,selection_mean_costs=means,held_cost_gain=cost(control)-cost(current),
                held_incoming_damage_reduction=control['received_health_damage']-current['received_health_damage'],
                held_outgoing_damage_loss=control['monster_health_damage']-current['monster_health_damage'])
        held.append(row)
    good=[r for r in held if r['known']];technical=all(r['strict_gate_passed'] for r in units)
    mean_gain=statistics.mean(r['held_cost_gain'] for r in good) if good else None
    mean_loss=statistics.mean(r['held_outgoing_damage_loss'] for r in good) if good else None
    useful=sum(r['held_cost_gain']>=2 for r in good)
    supported=bool(technical and len(good)>=signal['minimum_complete_worlds'] and
        useful>=signal['minimum_held_worlds_with_cost_gain_at_least_2hp'] and mean_gain>=signal['minimum_mean_held_cost_gain_hp'] and
        mean_loss<=signal['maximum_mean_held_outgoing_damage_loss_hp'])
    summary=dict(known_worlds=len(good),held_worlds_gain_at_least_2hp=useful,mean_held_cost_gain=mean_gain,mean_held_outgoing_damage_loss=mean_loss)
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True)
    for name,value in (('records',records),('preferences',preferences),('held-stream',held)):save(a.out/(name+'.json'),value)
    report=dict(state='complete',version='combat_underfire_horizon_pilot_report_v1',gate_passed=technical,pilot_signal_supported=supported,
        gate_scope='Strict native controls only; pilot signal separately gates scaling, never model/runtime admission.',units=units,summary=summary,
        artifacts={name:sha(a.out/(name+'.json')) for name in ('records','preferences','held-stream')},inputs=pins,
        training_performed=False,fit_admission=False,runtime_action_admitted=False,ordinary_training_eligible=False,promotion=False,
        scope='Four independent new TRAIN worlds, dependent branches and three policy continuations each. First2/third-stream pilot, not final quality or expected-Q proof. Unknown horizons masked; original strict failures retained.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(technical_gate=technical,pilot_signal_supported=supported,summary=summary,held=held))


if __name__=='__main__':main()
