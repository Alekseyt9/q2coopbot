"""Native motor-ablation prefix, observed descent and full episode receipts."""
import argparse
from pathlib import Path
from audit_combat_contact_loss_probe import load_steps
from audit_combat_joint_aim_grid import physical_equal_with_entity_bijection
from audit_combat_memory_route_zero import canonical_metrics
from evaluate_registered_combat_models import read,save,sha


def route_active(x):
    return bool(len(x)==1554 and x[0]>0 and x[1489]==1 and not any(x[73+12*j] for j in range(8))
        and any(x[881+30*j]==1 and x[1490+8*j]==1 and x[1491+8*j]==1 for j in range(8)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--screen',required=True,type=Path);ap.add_argument('--prepared',required=True,type=Path)
    ap.add_argument('--prior-screen',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args();assert not a.out.exists();root=a.screen.resolve()
    spec=read(root/'spec.json');pool=read(root/'pool/report.json');quality=read(root/'acceptance.json');proof=read(a.prepared/'report.json')
    assert quality['state']=='passed' and quality['numerical_verification_complete']
    assert pool['state']=='complete' and pool['source_unchanged'] and proof['changed_fields']==['motor_residual']
    jobs={j['plan_index']:j for j in pool['jobs']};groups=[];pins={}
    def pin(p):p=Path(p).resolve();pins[str(p)]=sha(p);return p
    for p in (root/'spec.json',root/'acceptance.json',root/'pool/report.json',a.prepared/'report.json'):pin(p)
    known_control=None
    for ci in range(spec['independent_worlds']):
        variants=[v for v in spec['variants'] if v['context_index']==ci];members={};native={};features={}
        for v in variants:
            capture=Path(jobs[v['index']]['root']);r=read(pin(capture/'report.json'))['results'][0]
            assert r['generated_fixture']==v['context']['instance']
            trace=pin(Path(r['root'])/'bot.jsonl')
            observations={}
            for b in load_steps(trace):
                o=(b.get('combat_policy')or{}).get('observation')or{}
                if o and o['identity']['life']==1 and o['health']>0 and o['identity']['frame']>=98:
                    observations[o['identity']['frame']]=o
            start=v['context']['instance']['player'];fall=next((dict(frame=f,position=o['position']) for f,o in sorted(observations.items())
                if o['position'][2]<start[2]-32),None)
            members[v['label']]=dict(goal=bool(r['goal_stop']),death=bool(r['death_stop']),initial_position=start,
                first_observed_lower_floor=fall,minimum_observed_z=min(o['position'][2] for o in observations.values()))
            if v['label']!='rules':
                d=root/f'case-{v["index"]:02d}';path=pin(next((d/'native').glob('replay-*/steps.jsonl')))
                native[v['label']]={s['observation']['identity']['frame']:s for s in load_steps(path) if s['owner']=='provider' and s['observation']['identity']['life']==1}
                rows=load_steps(pin(d/'cuda/rollout.jsonl'));features[v['label']]={row['frame']:row['features'] for row in rows}
            if v['label']=='source' and v['context']['seed']==1172325048:known_control=(v,r)
        left,right=native['source'],native['zero'];frames=sorted(set(left)&set(right));assert frames
        forward,reverse={},{};first=None;prefix=0
        for frame in frames:
            x,y=left[frame],right[frame]
            if x['sent_command']!=y['sent_command']:
                assert physical_equal_with_entity_bijection(x['observation'],y['observation'],forward,reverse), 'Different input before first changed command'
                fields=[k for k in x['sent_command'] if x['sent_command'][k]!=y['sent_command'][k]]
                first=dict(frame=frame,command_fields=fields,source_command=x['sent_command'],restored_command=y['sent_command'],
                    source_route_active=route_active(features['source'][frame]),restored_route_active=route_active(features['zero'][frame]))
                break
            assert physical_equal_with_entity_bijection(x['observation'],y['observation'],forward,reverse)
            prefix+=1
        assert first is not None,'No observed intervention effect'
        cutoff=members['source']['first_observed_lower_floor']
        cutoff=cutoff['frame'] if cutoff else 115
        groups.append(dict(seed=variants[0]['context']['seed'],members=members,first_command_divergence=first,identical_physical_command_prefix_rows=prefix,
            source_route_active_before_lower_floor=sum(route_active(x) for f,x in features['source'].items() if f<cutoff),
            restored_route_active_before_lower_floor=sum(route_active(x) for f,x in features['zero'].items() if f<cutoff)))
    # Reproduction of the known control is checked against the original wide run.
    assert known_control is not None
    old_spec=read(pin(a.prior_screen/'spec.json'));old_pool=read(pin(a.prior_screen/'pool/report.json'))
    old_v=next(v for v in old_spec['variants'] if v['label']=='source' and v['context']['seed']==1172325048)
    old_job=next(j for j in old_pool['jobs'] if j['plan_index']==old_v['index'])
    old_result=read(pin(Path(old_job['root'])/'report.json'))['results'][0]
    assert old_v['model_sha256']==known_control[0]['model_sha256'] and old_v['context']['instance']==known_control[0]['context']['instance']
    assert canonical_metrics(old_result)==canonical_metrics(known_control[1]),'Known control did not reproduce'
    save(a.out,dict(version='combat_motor_restore_causal_diagnostic_v1',state='passed',groups=groups,
        known_control_reproduced=True,changed_model_fields=['motor_residual'],inputs=pins,training_performed=False,promotion=False,
        scope='One known regression and three new seeds at one site. Motor weights also influence conditional aim/fire; not isolated motion or population/campaign quality.'))
    print([dict(seed=g['seed'],source_goal=g['members']['source']['goal'],restored_goal=g['members']['zero']['goal'],
                first_command_change=g['first_command_divergence']['frame']) for g in groups],flush=True)


if __name__=='__main__':main()
