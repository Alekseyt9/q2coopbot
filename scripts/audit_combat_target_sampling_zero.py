"""Strict full-trajectory source/zero native comparison, recorded metadata only."""
import argparse
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from audit_combat_joint_aim_grid import physical_equal_with_entity_bijection


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('probe','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--context-index',type=int,required=True)
    a=ap.parse_args();assert not a.out.exists();root=a.probe.resolve();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or h==expected,str(path);pins[key]=h;return path
    seal=read(pin(root/'complete.json'));assert seal['version']=='combat_native_target_sampling_zero_v1' and seal['native_cases']==seal['cuda_cases']==8
    spec=read(pin(root/'spec.json',seal['spec_sha256']));unit=spec['contexts'][a.context_index];c=unit['context']
    prep=read(pin(Path(unit['prepared'])/'report.json',unit['prepared_report_sha256']))
    assert prep['context']==c and prep['zero_source_raw_exact']
    pool=read(pin(root/'pool/report.json',seal['pool_sha256']));assert pool['source_unchanged']
    jobs={j['plan_index']:j for j in pool['jobs']};batch=read(pin(root/'cuda-batch/report.json',seal['cuda_receipt_sha256']))
    assert batch['state']=='complete' and batch['device']=='cuda'
    def load(folder,capture,expected_model=None):
        meta=read(pin(folder/'cuda/report.json'));assert meta['numerical_verification']=='cuda_verified_v1' and meta['observation_timing_contract']=='frozen_world_v1'
        if expected_model:assert meta['model_sha256']==expected_model
        proof=read(pin(folder/'cuda/cuda-verification.json',meta['cuda_verification_sha256']));assert proof['state']=='passed' and proof['device']=='cuda'
        sp=next((folder/'native').glob('replay-*/steps.jsonl'))
        all_steps=list(map(json.loads,pin(sp,meta['source_sha256'][str(sp.resolve())]).open(encoding='utf-8-sig')))
        steps=[s for s in all_steps if s['owner']=='provider' and s.get('sample') and s.get('native_step') and s['observation']['identity']['life']==1 and s['observation']['health']>0]
        rows={r['frame']:r for r in map(json.loads,pin(folder/'cuda/rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig'))}
        report=read(pin(capture));assert report['capture_complete'] and report['provenance_valid']
        return dict(steps=steps,rows=rows,result=report['results'][0],meta=meta)
    baseline=Path(prep['baseline']);bs=read(pin(baseline/'complete.json'))
    bp=read(pin(baseline/'pool/report.json',bs['pool_sha256']));old=next(j for j in bp['jobs'] if j['job']==c['job'])
    original=load(Path(c['directory']),Path(old['root'])/'report.json',prep['source_weights_sha256'])
    captures={}
    for v in unit['variants']:
        i=v['plan_index'];m=read(pin(v['model'],v['model_sha256']))
        if v['label']=='zero':assert m['target_sampling']['version']=='combat_target_temperature_v1' and m['target_sampling']['temperature']==1
        else:assert not m.get('target_sampling')
        captures[v['label']]=load(root/f'case-{i:02d}',Path(jobs[i]['root'])/'report.json',v['model_sha256'])
        assert captures[v['label']]['result']['generated_fixture']==c['instance']
    def compare(a,b):
        forward={};reverse={};left,right=a['steps'],b['steps'];mismatch=None
        fields=('sampling_seed','latent','attack','vertical','weapon','target','aim_mode','log_probability','value','memory','aim_candidate')
        for x,y in zip(left,right):
            f=x['observation']['identity']['frame']
            checks=dict(frame=f==y['observation']['identity']['frame'],
                observation=physical_equal_with_entity_bijection(x['observation'],y['observation'],forward,reverse),
                sample=all(x['sample'].get(k,0)==y['sample'].get(k,0) for k in fields),
                command=x['sent_command']==y['sent_command'],action=x['applied_action']==y['applied_action'],
                features=(f in a['rows'])==(f in b['rows']) and (f not in a['rows'] or a['rows'][f]['features']==b['rows'][f]['features']))
            nx,ny=x.get('next_observation'),y.get('next_observation')
            checks['next_observation']=(nx is None and ny is None) or (nx is not None and ny is not None and physical_equal_with_entity_bijection(nx,ny,forward,reverse))
            if not all(checks.values()):mismatch=dict(frame=f,failed=[k for k,v in checks.items() if not v]);break
        exact=len(left)==len(right) and mismatch is None
        return dict(exact=exact,left_steps=len(left),right_steps=len(right),first_mismatch=mismatch)
    source_check=compare(captures['control'],original);zero_check=compare(captures['zero'],captures['control'])
    active=len(captures['zero']['steps'])
    passed=source_check['exact'] and zero_check['exact'] and active>0
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True);report=dict(state='complete',version='combat_target_sampling_native_zero_acceptance_v1',context=c,
        gate_passed=passed,source_vs_archived_full=source_check,zero_vs_fresh_source_full=zero_check,checked_provider_rows=active,
        inputs=pins,neural_execution=False,training_performed=False,quality_evaluation=False,promotion=False,
        scope='Strict physical/command/sample/feature full first-life replay; model version string differs intentionally. No tail equivalence relaxation or combat quality claim.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(gate_passed=passed,map=c['instance']['map'],active_rows=active,source=source_check,zero=zero_check))


if __name__=='__main__':main()
