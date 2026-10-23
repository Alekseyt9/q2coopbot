"""Native common-state/draw acceptance and observed consequences; no NN calls."""
import argparse
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from audit_combat_joint_aim_grid import physical_equal_with_entity_bijection


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('probe','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--context-index',type=int,help='Audit one context from a sealed multi-context native pool')
    a=ap.parse_args();assert not a.out.exists();root=a.probe.resolve();pins={}

    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or expected==h,str(path)
        pins[str(path)]=h;return path

    seal=read(pin(root/'complete.json'));assert seal['version'] in ('combat_native_movement_probe_v1','combat_native_movement_pairs_v1')
    whole_spec=read(pin(root/'spec.json',seal['spec_sha256']))
    multi=seal['version']=='combat_native_movement_pairs_v1'
    assert (a.context_index is not None)==multi
    spec=whole_spec['contexts'][a.context_index] if multi else whole_spec
    context=spec['context'];frame=context['frame']
    prepared=read(pin(Path(spec['prepared'])/'report.json',spec['prepared_report_sha256']))
    assert prepared['context']==context
    assert context['split']=='train' or (multi and context['split']=='development' and prepared['collection_protocol'] and prepared['internal_development'])
    pool=read(pin(root/'pool/report.json',seal['pool_sha256']));jobs={j['plan_index']:j for j in pool['jobs']}
    assert pool['source_unchanged'] and len(jobs)==seal['native_cases']==seal['cuda_cases']
    batch=read(pin(root/'cuda-batch/report.json',seal['cuda_receipt_sha256']));assert batch['state']=='complete' and batch['device']=='cuda'
    source_path=pin(Path(prepared['source'])/'weights.json',prepared['source_weights_sha256'])
    source=read(source_path);source_canonical=json.dumps(source,sort_keys=True,separators=(',',':'))
    window_smoke=prepared.get('window_smoke',False)

    def load(directory,capture_path):
        meta=read(pin(directory/'cuda/report.json'));proof=read(pin(directory/'cuda/cuda-verification.json',meta['cuda_verification_sha256']))
        if directory.parent.resolve()==root:
            task=next(t for t in batch['tasks'] if Path(t['out']).resolve()==(directory/'cuda').resolve())
            assert sha(directory/'cuda/report.json')==task['report_sha256']
        assert proof['state']=='passed' and proof['device']=='cuda'
        sp=next((directory/'native').glob('replay-*/steps.jsonl'));pin(sp,meta['source_sha256'][str(sp.resolve())])
        steps=[json.loads(line) for line in sp.open(encoding='utf-8-sig')]
        controlled=[s for s in steps if s['owner']=='provider' and s.get('sample') and s.get('native_step') and s['observation']['identity']['life']==1 and s['observation']['health']>0 and (s.get('server_execution') or {}).get('matched') and (s.get('server_execution') or {}).get('window_exclusive') and (s.get('server_execution') or {}).get('recovery_commands')==0]
        rows=[json.loads(line) for line in pin(directory/'cuda/rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig')]
        rewards={r['step']:r for r in map(json.loads,pin(sp.with_name('rewards.jsonl'),meta['source_sha256'][str(sp.with_name('rewards.jsonl').resolve())]).open(encoding='utf-8-sig'))}
        capture=read(pin(capture_path));assert capture['capture_complete'] and capture['provenance_valid']
        return dict(meta=meta,proof=proof,steps=controlled,all_steps=steps,rows={r['frame']:r for r in rows},rewards=rewards,result=capture['results'][0])

    baseline=Path(prepared['baseline']);bc=read(pin(baseline/'complete.json'));bp=read(pin(baseline/'pool/report.json',bc['pool_sha256']))
    original_job=next(j for j in bp['jobs'] if j['job']==context['job'])
    reference_index=next((i for i,v in enumerate(spec['variants']) if v['label']=='ordinary_reference'),None)
    original=(load(root/f'case-{reference_index:02d}',Path(jobs[reference_index]['root'])/'report.json') if reference_index is not None
        else load(Path(context['directory']),Path(original_job['root'])/'report.json'))
    if reference_index is not None:assert original['meta']['observation_timing_contract']=='frozen_world_v1'
    control_index=spec['variants'][0]['plan_index'] if multi else 0
    control=load(root/f'case-{control_index:02d}',Path(jobs[control_index]['root'])/'report.json')
    if multi: assert control['meta']['observation_timing_contract']==original['meta']['observation_timing_contract']=='frozen_world_v1'

    def same_step(a,b,da,db,forward,reverse):
        sa,sb=a['sample'],b['sample'];f=a['observation']['identity']['frame']
        if f!=b['observation']['identity']['frame']:return False
        if not physical_equal_with_entity_bijection(a['observation'],b['observation'],forward,reverse):return False
        fields=('sampling_seed','latent','attack','vertical','weapon','target','aim_mode','memory','aim_candidate')
        if any(sa.get(k,0)!=sb.get(k,0) for k in fields):return False
        if a['sent_command']!=b['sent_command']:return False
        if (f in da['rows'])!=(f in db['rows']):return False
        if f in da['rows'] and da['rows'][f]['features']!=db['rows'][f]['features']:return False
        na,nb=a.get('next_observation'),b.get('next_observation')
        if (na is None)!=(nb is None):return False
        return na is None or physical_equal_with_entity_bijection(na,nb,forward,reverse)

    forward={};reverse={}
    full_control=len(control['steps'])==len(original['steps']) and all(same_step(x,y,control,original,forward,reverse) for x,y in zip(control['steps'],original['steps']))
    assert control['result']['generated_fixture']==original['result']['generated_fixture']==context['instance']
    control_at=next(s for s in control['steps'] if s['observation']['identity']['frame']==frame)
    draw=control_at['sample']['movement_probe'];assert draw['mean_shift']==[0.,0.]
    gamma=.99;results=[]
    for index,v in enumerate(spec['variants']):
        if multi:index=v['plan_index']
        if v['label']=='ordinary_reference':
            assert v['model_sha256']==prepared['source_weights_sha256']
            continue
        m=read(pin(v['model'],v['model_sha256']));probe=m.pop('diagnostic_movement_probe')
        # Serialization identity only; neural-state equality was checked on CUDA.
        assert json.dumps(m,sort_keys=True,separators=(',',':'))==source_canonical
        assert probe['frame']==frame and probe['mean_shift']==v['mean_shift']
        d=control if index==control_index else load(root/f'case-{index:02d}',Path(jobs[index]['root'])/'report.json')
        assert d['meta']['version']=='combat_diagnostic_movement_probe_rollout_v1' and d['meta']['diagnostic_only'] and not d['meta']['ppo_training_ready']
        assert d['proof']['diagnostic_fire_draw_verified'] and not d['proof']['ordinary_training_eligible']
        if reference_index is not None or multi:assert d['meta']['observation_timing_contract']=='frozen_world_v1'
        prefix=[s for s in d['steps'] if s['observation']['identity']['frame']<frame]
        reference=[s for s in control['steps'] if s['observation']['identity']['frame']<frame]
        forward={};reverse={}
        prefix_exact=len(prefix)==len(reference) and all(same_step(x,y,d,control,forward,reverse) for x,y in zip(prefix,reference))
        current=next(s for s in d['steps'] if s['observation']['identity']['frame']==frame)
        state_exact=(physical_equal_with_entity_bijection(current['observation'],control_at['observation'],forward,reverse)
            and d['rows'][frame]['features']==control['rows'][frame]['features']
            and current['sample']['aim_candidate']['timing']==control_at['sample']['aim_candidate']['timing'])
        receipt=current['sample']['movement_probe'];same_noise=receipt['gaussian_noise']==draw['gaussian_noise'] and receipt['fire_uniform']==draw['fire_uniform']
        window_draws_exact=True;window_receipts=0
        if window_smoke:
            duration=probe.get('duration_frames',1)
            assert duration==v['duration']
            frames={s['observation']['identity']['frame']:s for s in d['steps']}
            controls={s['observation']['identity']['frame']:s for s in control['steps']}
            observed=sorted(f for f in frames if frame<=f<frame+duration)
            window_receipts=len(observed)
            assert observed==list(range(frame,min(frame+duration,max(frames)+1))), 'Skipped intervention frame'
            for f in observed:
                r=frames[f]['sample'].get('movement_probe')
                assert r and r['version']==probe['version'] and r['frame']==f and r['mean_shift']==probe['mean_shift']
                if f in controls:
                    c=controls[f]['sample']['movement_probe']
                    window_draws_exact &= r['gaussian_noise']==c['gaussian_noise'] and r['fire_uniform']==c['fire_uniform']
        same_categories=all(current['sample'].get(k,0)==control_at['sample'].get(k,0) for k in ('target','aim_mode','vertical','weapon','memory'))
        source_move=lambda s:s['sample']['aim_candidate']['source_latent'][:2]
        a_move,b_move=source_move(current),source_move(control_at)
        shift_error=max(abs((x-y)-delta) for x,y,delta in zip(a_move,b_move,v['mean_shift']))
        assert shift_error<1e-10,shift_error
        r=d['result'];life=r['first_life'];damage=life['damage']
        task_end=r['goal_stop']['kill_frame'] if r['goal_stop'] else life['end_frame']
        rules_before_end=sum(s['owner']=='rules' and frame<=s['observation']['identity']['frame']<task_end and s['observation']['health']>0 for s in d['all_steps'])
        admitted=bool(full_control and prefix_exact and state_exact and same_noise and same_categories and window_draws_exact and rules_before_end==0)
        suffix=[row for f,row in d['rows'].items() if f>=frame]
        components=[(row,d['rewards'][row['index']]['components']) for row in suffix]
        observed=sum(gamma**(row['frame']-frame)*row['reward'] for row in suffix)
        sparse=sum(gamma**(row['frame']-frame)*(c.get('monster_kill',0)+c.get('death',0)) for row,c in components)
        results.append(dict(label=v['label'],admitted=admitted,baseline_full_exact=full_control,prefix_exact=prefix_exact,
            intervention_state_exact=state_exact,gaussian_and_fire_draws_exact=same_noise,unmodified_categories_exact=same_categories,
            movement_latent_shift_error=shift_error,goal=bool(r['goal_stop']),death=bool(r['death_stop']),
            window_receipts=window_receipts,window_common_draws_exact=window_draws_exact,
            kills=sum(q['monster_kills'] for q in damage['by_mod']),dealt=sum(q['monster_health_damage'] for q in damage['by_mod']),
            received=damage['received_health_damage'],observed_provider_return_no_bootstrap=observed,
            observed_provider_sparse_return_no_bootstrap=sparse,provider_suffix_rows=len(suffix),
            rules_commands_after_intervention=sum(s['owner']=='rules' and s['observation']['identity']['frame']>=frame for s in d['all_steps']),
            rules_commands_before_task_end=rules_before_end,
            gaussian_reconstruction_error=d['proof']['diagnostic_gaussian_noise_max_error']))
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True)
    report=dict(state='complete',version='combat_native_movement_probe_acceptance_v1',neural_execution=False,new_games=False,
        context=context,source_weights_sha256=prepared['source_weights_sha256'],baseline_full_exact=full_control,
        reference_basis='fresh_ordinary_frozen_world' if reference_index is not None or multi else 'historical_ordinary',
        admitted=sum(r['admitted'] for r in results),excluded=sum(not r['admitted'] for r in results),results=results,inputs=pins,
        training_performed=False,ordinary_training_eligible=False,promotion=False,
        scope='One training-registry context with shared draws; internal development is held out from fitting. Observed finite continuation, not expected Q or general battle quality. Only admitted variants can become labels.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(admitted=report['admitted'],excluded=report['excluded'],baseline_full_exact=full_control,results=results))


if __name__=='__main__':main()
