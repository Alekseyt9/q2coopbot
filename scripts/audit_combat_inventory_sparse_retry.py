"""Compare native first-life commands and effects with an intentionally lost packet."""
import argparse
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from audit_combat_joint_aim_grid import physical_equal_with_entity_bijection

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('control','drop','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path):
        path=Path(path).resolve();pins[str(path)]=sha(path);return path
    pin(__file__);reports=[read(pin(p/'report.json')) for p in (a.control,a.drop)];manifests=[read(pin(p/'manifest.json')) for p in (a.control,a.drop)]
    assert all(r['capture_complete'] and r['provenance_valid'] for r in reports)
    assert reports[0]['inventory_drop_once_frame']==0 and reports[1]['inventory_drop_once_frame']>0 and reports[1]['ordinary_training_eligible'] is False
    for key in ('source_fingerprint','native_source_fingerprint','model_weights_sha256','client_sha256','observation_timing_contract','generated_fixtures_sha256','reward_config_sha256'):
        assert manifests[0][key]==manifests[1][key],key
    assert manifests[0]['observation_timing_contract']=='frozen_world_v1';frame=reports[1]['inventory_drop_once_frame']
    byseed=[{r['seed']:r for r in report['results']} for report in reports];assert set(byseed[0])==set(byseed[1]);cases=[]
    for seed in sorted(byseed[0]):
        pair=[group[seed] for group in byseed];loaded=[]
        for result in pair:
            assert result['dispatch_valid'];folder=Path(result['root']);log=pin(folder/'server.log')
            drops=[line for line in log.open(encoding='utf-8-sig') if line.startswith('sv_test_inventory_drop ')]
            cutoff=result['goal_stop']['kill_frame'] if result['goal_stop'] else result['first_life']['end_frame']
            path=pin(folder/'dataset/steps.jsonl');steps=[s for s in map(json.loads,path.open(encoding='utf-8-sig')) if s['owner']=='provider' and s.get('sample') and s.get('native_step') and s['observation']['identity']['life']==1 and s['observation']['health']>0 and 100<=s['observation']['identity']['frame']<cutoff]
            rows={s['observation']['identity']['frame']:s for s in steps};assert len(rows)==len(steps)
            assert frame in rows or (frame==cutoff and any(s.get('next_observation') and s['next_observation']['identity']['frame']==frame for s in steps)), 'Lost packet must be an observed alive frame or observed task/death endpoint'
            effects={r['step']:r for r in map(json.loads,pin(folder/'dataset/server_outcomes.jsonl').open(encoding='utf-8-sig'))}
            loaded.append(dict(rows=rows,effects=effects,drops=drops,cutoff=cutoff))
        left,right=loaded;assert len(left['drops'])==0 and len(right['drops'])==1 and f'frame={frame} ' in right['drops'][0]
        assert left['cutoff']==right['cutoff'] and set(left['rows'])==set(right['rows'])
        forward={};reverse={};count=0
        for f in sorted(left['rows']):
            x,y=left['rows'][f],right['rows'][f]
            assert physical_equal_with_entity_bijection(x['observation'],y['observation'],forward,reverse),(seed,f,'observation')
            assert physical_equal_with_entity_bijection(x['next_observation'],y['next_observation'],forward,reverse),(seed,f,'next_observation')
            assert x['sample']==y['sample'],(seed,f,'sample')
            assert x['sent_command']==y['sent_command'] and x['applied_action']==y['applied_action'],(seed,f,'commands')
            assert x['observation']['inventory_age_frames']==y['observation']['inventory_age_frames']==0
            ex,ey=left['effects'][x['index']],right['effects'][y['index']]
            assert ex['available'] and ey['available']
            for key in ('monster_health_damage','received_health_damage','self_health_damage','teammate_health_damage','monster_kills','deaths'):assert ex[key]==ey[key],(seed,f,key)
            count+=1
        cases.append(dict(seed=seed,provider_frames=count,drop_frame=frame,inventory_age_frames=0,first_life_exact=True))
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True);save(a.out/'report.json',dict(state='complete',version='combat_inventory_sparse_retry_native_audit_v1',gate_passed=True,cases=cases,inputs=pins,neural_execution=False,new_games=False,training_performed=False,
        scope='Native first-life provider observations, samples, commands and aggregate per-step effects through task/death/budget boundary. Does not assert equality of post-task capture tails or prove general combat quality. A dropped endpoint packet is verified through the preceding alive step next_observation. GPU numerical replay is a separate check.'))
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')));print(cases)

if __name__=='__main__':main()
