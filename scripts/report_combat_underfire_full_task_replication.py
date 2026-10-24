"""Evaluate frozen movement choices by complete fight success on new streams."""
import argparse
from pathlib import Path
import statistics
from evaluate_registered_combat_models import read,save,sha
from select_combat_underfire_full_task import utility


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();root=a.collection.resolve();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or h==expected,str(path);pins[key]=h;return path
    pin(__file__);pin(Path(__file__).with_name('select_combat_underfire_full_task.py'))
    closed=read(pin(root/'complete.json'));assert closed['version']=='combat_native_continuation_pairs_v1'
    spec=read(pin(root/'spec.json',closed['spec_sha256']));p=read(pin(spec['protocol'],spec['protocol_sha256']))
    assert p['study']=='combat_underfire_full_task_replication_v1' and closed['native_cases']==closed['cuda_cases']==37
    sr=Path(p['selection_out']);ss=read(pin(sr/'complete.json'))
    selection=read(pin(sr/'report.json',ss['report_sha256']));assert selection['protocol_sha256']==sha(spec['protocol'])
    for path,h in selection['inputs'].items():pin(path,h)
    choices={u['context']['context_id']:u for u in selection['units']};assert len(choices)==4
    parent=Path(p['parent_pilot']);ps=read(pin(parent/'dataset/complete.json'))
    original=read(pin(parent/'dataset/report.json',ps['report_sha256']));assert not original['pilot_signal_supported']
    assert p['continuation_seeds']==[2026102406,2026102407,2026102408]
    units=[];groups={cid:[] for cid in choices}
    for unit in spec['contexts']:
        c=unit['context'];choice=choices[c['context_id']];assert choice['context']==c and c['role']=='auxiliary_train'
        folder=root/'proofs'/unit['unit_id'];seal=read(pin(folder/'complete.json'))
        proof=read(pin(folder/'report.json',seal['report_sha256']))
        for path,h in proof['inputs'].items():pin(path,h)
        assert proof['context']==c and proof['fork']['seed'] in p['continuation_seeds']
        results={r['label']:r for r in proof['results']};assert set(results)=={'control',choice['selected']}
        control,current=(results[label] for label in ('control',choice['selected']))
        known=proof['gate_passed'] and all(r['goal'] or r['death'] for r in (control,current))
        row=dict(unit_id=unit['unit_id'],context_id=c['context_id'],map=c['instance']['map'],stream_seed=proof['fork']['seed'],
            selected=choice['selected'],strict_gate_passed=proof['gate_passed'],terminal_known=known,control=control,current=current)
        if known:
            a_goal,b_goal=(bool(r['goal'] and not r['death']) for r in (control,current))
            row.update(gained_alive_goal=int(b_goal and not a_goal),lost_alive_goal=int(a_goal and not b_goal),
                control_alive_goal=int(a_goal),selected_alive_goal=int(b_goal),utility_gain=utility(current)-utility(control),
                kill_gain=current['kills']-control['kills'],received_reduction=control['received']-current['received'])
        units.append(row);groups[c['context_id']].append(row)
    assert len(units)==12
    worlds=[]
    for cid,rows in groups.items():
        assert len(rows)==3 and {r['stream_seed'] for r in rows}==set(p['continuation_seeds'])
        known=all(r['terminal_known'] for r in rows);world=dict(context=choices[cid]['context'],selected=choices[cid]['selected'],known=known)
        if known:
            world.update({field:statistics.mean(r[field] for r in rows) for field in ('utility_gain','kill_gain','received_reduction')})
            world.update({field:sum(r[field] for r in rows) for field in ('gained_alive_goal','lost_alive_goal','control_alive_goal','selected_alive_goal')})
        worlds.append(world)
    good=[r for r in units if r['terminal_known']];good_worlds=[r for r in worlds if r['known']]
    gained=sum(r['gained_alive_goal'] for r in good);lost=sum(r['lost_alive_goal'] for r in good)
    mean_gain=statistics.mean(r['utility_gain'] for r in good_worlds) if good_worlds else None
    positive_worlds=sum(r['utility_gain']>=.05 for r in good_worlds);technical=all(r['strict_gate_passed'] for r in units);signal=p['signal']
    supported=bool(technical and len(good)>=signal['minimum_terminal_known_world_stream_pairs'] and
        gained-lost>=signal['minimum_net_gained_alive_goals'] and lost<=signal['maximum_lost_alive_goals'] and
        positive_worlds>=signal['minimum_worlds_mean_utility_gain_at_least_0_05'] and mean_gain>=signal['minimum_mean_utility_gain'])
    summary=dict(terminal_known_pairs=len(good),known_worlds=len(good_worlds),gained_alive_goals=gained,lost_alive_goals=lost,
        control_alive_goals=sum(r['control_alive_goal'] for r in good),selected_alive_goals=sum(r['selected_alive_goal'] for r in good),
        worlds_utility_gain_at_least_0_05=positive_worlds,mean_utility_gain=mean_gain)
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True);save(a.out/'units.json',units);save(a.out/'worlds.json',worlds)
    report=dict(state='complete',version='combat_underfire_full_task_replication_report_v1',gate_passed=technical,signal_supported=supported,
        summary=summary,worlds=worlds,inputs=pins,units_sha256=sha(a.out/'units.json'),worlds_sha256=sha(a.out/'worlds.json'),
        original_local_proxy_failure_retained=True,training_performed=False,fit_admission=False,runtime_action_admitted=False,
        ordinary_training_eligible=False,promotion=False,
        scope='Three unseen policy-stream repeats in four fixed TRAIN worlds. Full task metrics for preselected per-root counterfactuals, not an implemented learned selector, independent world sample or superiority over rules.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(technical_gate=technical,signal_supported=supported,summary=summary,worlds=worlds))


if __name__=='__main__':main()
