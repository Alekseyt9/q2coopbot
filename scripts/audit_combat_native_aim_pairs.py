"""Seal native counterfactual TRAIN labels only for accepted common prefixes."""
import argparse
import collections
import json
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha
from audit_combat_joint_aim_context_prefix import category, source_latent
from audit_combat_joint_aim_grid import physical_equal_with_entity_bijection
from report_combat_machinegun_hits import scan
from report_registered_machinegun_outcomes import outcome


def same_draws(s,b):
    return (source_latent(s)==source_latent(b) and s.get('memory')==b.get('memory')
        and all(s.get(k,0)==b.get(k,0) for k in ('sampling_seed','target','aim_mode','vertical','weapon')))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('probe','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={};root=a.probe.resolve()
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    closed=read(pin(root/'complete.json'));assert closed['version']=='combat_native_aim_pairs_probe_v1' and closed['native_cases']==closed['cuda_cases']
    spec=read(pin(root/'spec.json',closed['spec_sha256']));assert spec['split']=='train' and not spec['training_performed']
    prepared=read(pin(Path(spec['prepared'])/'report.json',spec['prepared_report_sha256']))
    one_frame=prepared['version']=='combat_one_frame_aim_preparation_v1'
    count=len(spec['variants']);assert count==closed['native_cases']
    if one_frame:assert len(prepared['contexts'])==1 and count in (2,5)
    else:assert count==5*len(prepared['contexts'])
    baseline=Path(prepared['baseline']);baseclosed=read(pin(baseline/'complete.json'))
    basepool=read(pin(baseline/'pool/report.json',baseclosed['pool_sha256']));basejobs={j['job']:j for j in basepool['jobs']}
    pool=read(pin(root/'pool/report.json',closed['pool_sha256']));assert pool['state']=='complete' and pool['source_unchanged']
    jobs={j['plan_index']:j for j in pool['jobs']};assert set(jobs)==set(range(count))
    source=read(pin(Path(prepared['source'])/'weights.json',prepared['source_weights_sha256']))
    for v in spec['variants']:
        model=read(pin(v['model'],v['model_sha256']));assert model['aim_candidate_head']['layers'][0]==source['aim_candidate_head']['layers'][0]
        if one_frame:
            probe=model.pop('diagnostic_joint_aim_probe');assert probe['version']=='combat_joint_aim_one_frame_probe_v1' and probe['frame']==v['frame']
            assert model==source,'Neural parameters changed in metadata-only probe'
        model['aim_candidate_head']['layers']=None;s=json.loads(json.dumps(source));s['aim_candidate_head']['layers']=None;assert model==s
    for receipt in closed['cuda_receipts']:
        r=read(pin(receipt['report'],receipt['report_sha256']));assert r['state']=='complete' and r['device']=='cuda'
        for t in r['tasks']:pin(Path(t['out'])/'report.json',t['report_sha256'])
    def load(folder,job,frame):
        meta=read(pin(folder/'cuda/report.json'));verification=read(pin(folder/'cuda/cuda-verification.json'))
        assert verification['state']=='passed' and verification['device']=='cuda'
        if one_frame and root in folder.parents:
            assert meta['version']=='combat_diagnostic_joint_aim_probe_rollout_v1' and meta['diagnostic_only'] and not meta['ppo_training_ready']
        path=pin(folder/'native/replay-0/steps.jsonl');assert meta['source_sha256'][str(path)]==sha(path)
        rows=list(map(json.loads,pin(folder/'cuda/rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig')))
        byframe={r['frame']:r for r in rows};steps=[]
        for s in map(json.loads,path.open(encoding='utf-8-sig')):
            o=s['observation'];d=s.get('server_execution') or {}
            if (s['owner']=='provider' and s.get('sample') and s.get('native_step') and o['identity']['life']==1 and o['health']>0
                and d.get('matched') and d.get('window_exclusive') and d.get('recovery_commands')==0 and o['identity']['frame']<=frame):steps.append(s)
        assert steps and steps[-1]['observation']['identity']['frame']==frame
        capture=read(pin(Path(job['root'])/'report.json'));assert capture['capture_complete'] and capture['provenance_valid']
        result=capture['results'][0];assert result['capture_valid'] and result['dispatch_valid'] and result['seed_confirmed']
        assert result['seed']==job['seed'] and result['generated_fixture']['split']=='train'
        log=pin(Path(result['root'])/'server.log');n=steps[-1]['native_step'];key=(n['spawncount'],n['begin_frame'],n['actor'],n['sequence'])
        with log.open(encoding='utf-8-sig') as stream:shots=[q for q in scan(stream,{key}) if q['eligible']]
        return dict(steps=steps,rows=byframe,result=result,shots=shots)
    results=[];labels=[]
    for context in prepared['contexts']:
        ci=context['context_id'];frame=context['frame'];variants=[(i,v) for i,v in enumerate(spec['variants']) if v['context_id']==ci]
        bi=next(i for i,v in variants if v['category']==0);base=load(root/f'case-{bi:02d}',jobs[bi],frame)
        old=load(baseline/'processing'/f"case-{context['job']:04d}",basejobs[context['job']],frame)
        bs=base['steps'];os=old['steps'];history_exact=len(bs)==len(os) and all(
            same_draws(s['sample'],o['sample']) and s['sample']['latent']==o['sample']['latent'] and category(s['sample'])==category(o['sample'])
            and s['sample']['attack']==o['sample']['attack'] and s['sent_command']==o['sent_command']
            and base['rows'][s['observation']['identity']['frame']]['features']==old['rows'][o['observation']['identity']['frame']]['features'] for s,o in zip(bs,os))
        samples=[]
        for index,v in variants:
            current=base if index==bi else load(root/f'case-{index:02d}',jobs[index],frame);steps=current['steps'];issues=[];forward={};reverse={};matched=0
            if not history_exact:issues.append('Baseline failed to reproduce historical actions/features')
            if len(steps)!=len(bs):issues.append('Prefix length differs')
            if current['result']['generated_fixture']!=base['result']['generated_fixture']:issues.append('Fixture differs')
            for s,b in zip(steps,bs):
                f=s['observation']['identity']['frame'];ss=s['sample'];bb=b['sample']
                if not physical_equal_with_entity_bijection(s['observation'],b['observation'],forward,reverse):issues.append(f'Physical state differs at {f}');break
                if current['rows'][f]['features']!=base['rows'][f]['features'] or not same_draws(ss,bb):issues.append(f'Policy inputs/draws differ at {f}');break
                if f==frame:
                    if category(ss)!=v['category']:issues.append('Unexpected intervention category')
                    if (ss.get('aim_candidate') or {}).get('timing')!=(bb.get('aim_candidate') or {}).get('timing'):issues.append('Head timing differs at intervention')
                elif s['sent_command']!=b['sent_command'] or ss['latent']!=bb['latent'] or ss['attack']!=bb['attack'] or category(ss)!=0:
                    issues.append(f'Executed prefix differs at {f}');break
                else:matched+=1
            records=[];target_index=steps[-1]['sample']['target'];enemies=steps[-1]['observation']['enemies']
            selected=enemies[target_index-1] if 1<=target_index<=len(enemies) else None
            for shot in current['shots']:
                d=shot['damage'];f=shot['fields'];before=int(f['health_before']);damage=max(0,min(max(before,0),before-int(d['health_after']))) if d else 0
                records.append(dict(outcome=outcome(shot),health_damage=damage,selected_target_hit=bool(selected and selected['id']==int(f['target'])),fields=f,damage=d))
            if len(records)>1:issues.append('Multiple native MG shots in intervention window')
            spread_same=None
            if len(current['shots'])==len(base['shots'])==1:
                q=current['shots'][0]['fields'];b=base['shots'][0]['fields']
                # Water/muzzle paths may causally change spread generation. In
                # identical path classes, require the same actual seeded spread.
                if q['water']==b['water'] and q['muzzle_blocked']==b['muzzle_blocked']:
                    spread_same=q['spread']==b['spread']
                    if not spread_same:issues.append('Native spread differs for the same water/muzzle path')
            samples.append(dict(category=v['category'],accepted=not issues,issues=issues,matching_prefix_frames=matched,
                entity_id_bijection=forward,actual_spread_equal=spread_same,attack=steps[-1]['sample']['attack'],shots=records,
                native_outcome=records[0]['outcome'] if records else 'no_native_shot',health_damage=sum(q['health_damage'] for q in records)))
        accepted=[s for s in samples if s['accepted']];b=next(s for s in samples if s['category']==0)
        improvements=[s['category'] for s in accepted if s['health_damage']>b['health_damage']] if b['accepted'] else []
        regressions=[s['category'] for s in accepted if s['health_damage']<b['health_damage']] if b['accepted'] else []
        results.append(dict(context_id=ci,geometry=context['geometry'],seed=context['seed'],frame=frame,baseline_selection_outcome=context['baseline_outcome'],
            historical_actions_features_reproduced=history_exact,baseline_native_outcome_reproduced=b['native_outcome']==context['baseline_outcome'],
            accepted_alternatives=len(accepted),improving_categories=improvements,regressing_categories=regressions,samples=samples))
        if b['accepted']:
            row=base['rows'][frame];labels.append(dict(version='combat_native_one_frame_aim_probe_labels_v1' if one_frame and count==2 else 'combat_native_aim_pair_labels_v1',context_id=ci,geometry=context['geometry'],
                episode=context.get('episode',spec['variants'][bi]['episode']),seed=context['seed'],frame=frame,features=row['features'],sample=row['sample'],
                predictor_hit_probabilities=context.get('predicted_hit_probabilities'),baseline_health_damage=b['health_damage'],
                alternatives=[dict(category=s['category'],outcome=s['native_outcome'],health_damage=s['health_damage'],
                    delta_from_baseline=s['health_damage']-b['health_damage']) for s in accepted],
                scope='Native one-state TRAIN counterfactual labels, not an on-policy PPO transition. Group splits by context/geometry.'))
    for p,h in pins.items():assert sha(Path(p))==h,p
    a.out.mkdir(parents=True);dataset=a.out/'labels.jsonl';dataset.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in labels),encoding='utf-8')
    report=dict(state='complete',version='combat_native_one_frame_aim_probe_audit_v1' if one_frame else 'combat_native_aim_pairs_audit_v1',contexts=results,label_contexts=len(labels),
        one_frame_probe=one_frame,
        accepted_variants=sum(c['accepted_alternatives'] for c in results),total_variants=count,labels_sha256=sha(dataset),inputs=pins,
        improving_contexts=sum(bool(c['improving_categories']) for c in results),regressing_contexts=sum(bool(c['regressing_categories']) for c in results),
        script_sha256=sha(__file__),training_performed=False,validation_used=False,quality_confirmation=False,promotion=False,
        scope='Targeted multi-context TRAIN data with accepted physical/input/draw prefixes and native one-step labels. No population quality/generalization claim or fitting.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json'),labels_sha256=sha(dataset)))
    print(json.dumps(dict(accepted_variants=report['accepted_variants'],improving_contexts=report['improving_contexts'],regressing_contexts=report['regressing_contexts'],
        contexts=[dict(geometry=c['geometry'],historical_reproduced=c['historical_actions_features_reproduced'],
            improvements=c['improving_categories'],regressions=c['regressing_categories'],
            variants=[{k:s[k] for k in ('category','accepted','issues','native_outcome','health_damage')} for s in c['samples']]) for c in results]),indent=2))


if __name__=='__main__':main()
