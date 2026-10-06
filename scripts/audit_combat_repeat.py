import pathlib,json,hashlib,re,argparse
ap=argparse.ArgumentParser(description='Offline deterministic combat repeat audit; no training.')
ap.add_argument('root',type=pathlib.Path);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--require-equal',action='store_true')
ap.add_argument('--learned-prefix',action='store_true',help='Compare uninterrupted provider segment before first return to rules; no native outcome equality claim.')
args=ap.parse_args()
root=args.root
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
reports={p:read(root/p/'report.json') for p in ('first','repeat')}
manifests={p:read(root/p/'manifest.json') for p in reports}
for p,m in manifests.items():
    assert m['release_game_frame']==100 and m['post_frame_rng_reset'] and m['mixed'] and m['synchronous'] and m['timescale']==2
    assert sha(pathlib.Path(m['model_weights']))==m['model_weights_sha256'].lower()
    assert reports[p]['capture_complete'] and reports[p]['provenance_valid'] and reports[p]['parallelism']==4
    assert {e['seed'] for e in reports[p]['results']}==set(range(args.seed,args.seed+4))
for key in ('release_game_frame','model_weights_sha256','source_fingerprint','native_source_fingerprint','game_frames','reward_config_sha256'):
    assert manifests['first'][key]==manifests['repeat'][key],key
def difference(a,b,path=''):
    if type(a)!=type(b):return path
    if isinstance(a,dict):
        if a.keys()!=b.keys():return path+'.keys'
        for k in a:
            d=difference(a[k],b[k],path+'.'+k)
            if d:return d
    elif isinstance(a,list):
        if len(a)!=len(b):return path+'.length'
        for i,(x,y) in enumerate(zip(a,b)):
            d=difference(x,y,path+f'[{i}]')
            if d:return d
    elif a!=b:return path
    return None
records=[]
for first,repeat in zip(sorted(reports['first']['results'],key=lambda e:e['seed']),sorted(reports['repeat']['results'],key=lambda e:e['seed'])):
    pair=[]
    for ep in (first,repeat):
        assert all(ep[k] for k in ('capture_valid','dispatch_valid','seed_confirmed'))
        episode_root=pathlib.Path(ep['root'])
        log=(episode_root/'server.log').read_text()
        for pattern in (r'^g_test_world_start frame=100 phase=fixed_map_hold free_pool_reset=1$',r'^g_test_rng_start game_frame=100 phase=post_frame seed='+str(ep['seed'])+r' cursor_before=\d+ cursor_after=256$',r'^g_test_weapon_start game_frame=100 actor=1 weapon=Blaster gunframe_before=\d+ gunframe_after=9$'):
            assert len(re.findall(pattern,log,re.M))==1,('native reset receipt',ep['seed'],pattern)
        ds=episode_root/'dataset'
        steps=[json.loads(s) for s in (ds/'steps.jsonl').read_text().splitlines()]
        rewards=[json.loads(s) for s in (ds/'rewards.jsonl').read_text().splitlines()]
        assert steps and len(steps)==len(rewards)
        life=steps[0]['observation']['identity']['life'];rows=[];handoffs=[];kills=[]
        for s,r in zip(steps,rewards):
            if s['observation']['identity']['life']!=life:continue
            o=s['observation'].copy();o.pop('observation_age_ms',None)
            rows.append(dict(observation=o,action=s['applied_action'],owner=s['owner'],reason=s.get('reason')))
            if s.get('reason')=='control_handoff':handoffs.append(o['identity']['frame'])
            if r['available'] and r['components']['monster_kill']>0:
                assert s['owner']=='provider' and s['server_execution']['matched'] and s['server_execution']['window_exclusive']
                kills.append(o['identity']['frame'])
        outcome=dict(kills=sum(t['kills'] for t in ep['first_life']['outgoing_by_target_class']),received=ep['first_life']['damage']['received_health_damage'],end=ep['first_life']['end_reason'])
        pair.append(dict(rows=rows,outcome=outcome,handoffs=handoffs,kills=kills))
    def prefix(rows):
        out=[]
        for row in rows:
            if row['owner']=='provider':out.append({k:v for k,v in row.items() if k!='reason'})
            elif out:break
        assert out,'No learned prefix'
        return out
    compared=[prefix(p['rows']) if args.learned_prefix else p['rows'] for p in pair]
    diff=difference(*compared)
    firstdiv=next((dict(index=i,frame=a['observation']['identity']['frame'],field=difference(a,b)) for i,(a,b) in enumerate(zip(*compared)) if a!=b),None)
    records.append(dict(seed=first['seed'],compared_rows=[len(p) for p in compared],first_observation_equal=compared[0][0]['observation']==compared[1][0]['observation'],trajectory_equal=diff is None,full_trajectory_equal=pair[0]['rows']==pair[1]['rows'],outcome_equal=pair[0]['outcome']==pair[1]['outcome'],first_difference=firstdiv,first={k:v for k,v in pair[0].items() if k!='rows'},repeat={k:v for k,v in pair[1].items() if k!='rows'}))
out=dict(release_frame=100,comparison='learned_prefix' if args.learned_prefix else 'full_first_life',records=records,all_observed_trajectories_equal=all(r['trajectory_equal'] for r in records),scope='Compares observed state and applied commands excluding wall-clock observation age. Prefix mode excludes rules after handoff and does not assert whole-episode or native outcome equality. Does not prove complete native world or hidden AI state equality.')
(root/('prefix-repeat-audit.json' if args.learned_prefix else 'repeat-audit.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(comparison=out['comparison'],all_observed_trajectories_equal=out['all_observed_trajectories_equal'],records=[{k:v for k,v in r.items() if k not in ('first','repeat')} for r in records]),indent=2))
if args.require_equal:
    assert out['all_observed_trajectories_equal'],'Observed trajectories differ'
    if not args.learned_prefix:assert all(r['outcome_equal'] for r in records),'Native first-life outcomes differ'
