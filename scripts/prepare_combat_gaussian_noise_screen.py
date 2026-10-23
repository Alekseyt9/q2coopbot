"""Freeze fresh matched validation fixtures and the predeclared Gaussian-noise screen."""
import argparse
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from collect_combat_joint_aim_budget_round import historical_plans


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','movement','all','protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--port',type=int,default=46900)
    a=ap.parse_args();assert not a.out.exists();repo=Path(__file__).resolve().parents[1];pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or expected==h,str(path)
        pins[str(path)]=h;return path
    p=read(pin(a.protocol));assert p['version']=='combat_gaussian_noise_screen_v1'
    registry=pin(p['registry']);episodes=[read(pin(registry.parent/f)) for f in read(registry)['files']]
    assert len(episodes)==8 and {e['map'] for e in episodes}=={'base3','city1','jail1','ware2'}
    assert all(e['timescale']==2 and 'rules' in e['modes'] and 'learned' in e['modes'] for e in episodes)
    roots=dict(source=a.source.resolve(),**{'movement-quarter':a.movement.resolve(),'all-quarter':a.all.resolve()})
    for label,folder in roots.items():
        seal=read(pin(folder/'complete.json'))
        for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:pin(folder/name,seal[key])
        model=read(folder/'weights.json');report=read(folder/'report.json');assert report['device']=='cuda'
        assert not model['deterministic'] and not any(model.get(k) for k in ('diagnostic_joint_aim_probe','diagnostic_movement_probe','diagnostic_continuation_probe','execution_mode'))
        if label=='source':assert seal['weights_sha256']==p['source_weights_sha256']
        else:
            assert report['variant']==label and report['source_weights_sha256']==p['source_weights_sha256']
            assert report['checkpoint_roundtrip_verified'] and report['only_changed_std_moments_reset'] and report['critic_geometry_optimizer_rng_history_config_retention_exact']
            assert report['scale']==p['variants'][label] and not report['training_performed']
    history=historical_plans(repo/'workspace/artifacts');used={s for _,plan in history for task in plan['tasks'] for s in task.get('seeds',[])}
    offset=0
    while True:
        assert all(offset+1<=e['splits']['validation']['count'] for e in episodes)
        seeds={e['splits']['validation']['start']+offset for e in episodes}
        if not seeds&used:break
        offset+=1
    assert len(seeds)==8
    a.out.mkdir(parents=True)
    variants=[dict(label=label,mode='learned',update=str(folder),deterministic=False,cuda_replay=True,checked_wall_slide=True) for label,folder in roots.items()]
    variants.append(dict(label='rules',mode='rules',cuda_replay=False))
    comparisons=[dict(name=before+'-vs-'+after,before=before,after=after) for before,after in (
        ('source','movement-quarter'),('source','all-quarter'),('rules','movement-quarter'),('rules','all-quarter'),('movement-quarter','all-quarter'))]
    spec=dict(version='registered_combat_model_comparison_v1',registry=str(registry),split='validation',count=1,
        max_instances=16,episodes=sorted(e['id'] for e in episodes),seed_offset=offset,port=a.port,minimum_free_gib=32,
        ready=[str((a.out/'complete.json').resolve())],variants=variants,comparisons=comparisons,
        historical_plans=[str(path) for path,_ in history],unavailable_historical_references=historical_plans.missing_references,
        experiment_protocol=str(a.protocol.resolve()),experiment_protocol_sha256=sha(a.protocol),training_performed=False,promotion=False)
    save(a.out/'spec.json',spec)
    for path,h in pins.items():assert sha(path)==h,path
    report=dict(state='complete',version='combat_gaussian_noise_screen_preparation_v1',seed_offset=offset,seeds=sorted(seeds),
        native_cases=32,cuda_cases=24,inputs=pins,protocol_sha256=sha(a.protocol),spec_sha256=sha(a.out/'spec.json'),
        source_weights_sha256=p['source_weights_sha256'],training_performed=False,promotion=False)
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json'),spec_sha256=sha(a.out/'spec.json')))
    print(dict(native_cases=32,cuda_cases=24,split='validation',seed_offset=offset,seeds=sorted(seeds)))


if __name__=='__main__':main()
