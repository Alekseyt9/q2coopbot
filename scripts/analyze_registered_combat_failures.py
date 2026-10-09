"""Offline trace diagnostics; does not supply observations or control the game."""
import argparse, collections, hashlib, json, math, pathlib, re

def read(p):
    return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))

def wrap(degrees):
    return (degrees+180)%360-180

def analyze(root):
    root=pathlib.Path(root)
    rows=[json.loads(s) for s in (root/'bot.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    text=(root/'server.log').read_text(encoding='utf-8-sig')
    release=re.search(r'^sv_test_combat spawncount=(-?\d+) server_frame=(\d+) g_test_combat_start game_frame=100 ready=1 seed=(\d+)$',text,re.M)
    if not release: raise ValueError('Missing fixed release')
    generation,frame,seed=map(int,release.groups())
    actor=rows[0]['self_entity']; life=[]
    for row in rows:
        if row['spawncount']!=generation or row['map']!=rows[0]['map']: break
        if row['observation_frame']<=frame: continue
        life.append(row)
        if row['health']<=0: break
    seen=set();live=[]
    for row in life:
        if row['observation_frame'] not in seen:
            seen.add(row['observation_frame']);live.append(row)
    counters=collections.Counter();yaw_errors=[];distances=[];guards=collections.Counter();owners=collections.Counter();limits=collections.Counter()
    for i,row in enumerate(live):
        if row['health']<=0: continue
        obs=row['combat_policy']['observation'];cmd=row['sent_command'];sel=row['combat_policy']['selection']
        enemies=[e for e in obs['enemies'] if e.get('clear_shot')]
        counters['frames']+=1;owners[sel['owner']]+=1
        if row.get('arbitration',{}).get('limit_reason'): limits[row['arbitration']['limit_reason']]+=1
        firing=bool(cmd['Buttons']&1)
        counters['attack_frames']+=firing;counters['visible_frames']+=bool(enemies)
        counters['visible_attack_frames']+=bool(enemies) and firing
        counters['attack_without_clear_enemy']+=not enemies and firing
        if enemies:
            distances.append(min(e['distance'] for e in enemies))
            yaw=(cmd['Yaw']+row['delta_angles'][1])*360/65536
            errors=[abs(wrap(yaw-math.degrees(math.atan2(e['relative'][1],e['relative'][0])))) for e in enemies]
            if firing:
                yaw_errors.append(min(errors));counters['attack_yaw_gt_20']+=min(errors)>20
                counters['attack_yaw_lt_5']+=min(errors)<5
        counters['jump_frames']+=cmd['Up']>0;counters['crouch_frames']+=cmd['Up']<0
        for change in sel.get('interventions',[]):
            guards[change.get('reason','unknown')]+=1
        counters['command_move_frames']+=math.hypot(cmd['Forward'],cmd['Side'])>100
        if i+1<len(live):
            nxt=live[i+1]
            if nxt['observation_frame']==row['observation_frame']+1 and row.get('on_ground') and nxt.get('on_ground') and math.hypot(cmd['Forward'],cmd['Side'])>100:
                counters['ground_move_pairs']+=1
                counters['ground_move_under_1unit']+=math.dist(row['self'][:2],nxt['self'][:2])<1
    outgoing=collections.Counter();incoming=collections.Counter();hits=set();kills=0
    end=live[-1]['observation_frame']
    for line in text.splitlines():
        if not line.startswith('sv_test_damage '): continue
        fields=dict(re.findall(r'(\w+)=([^ ]+)',line))
        if int(fields['spawncount'])!=generation or not frame<int(fields['server_frame'])<=end: continue
        damage=min(max(0,int(fields['health_before'])),int(fields['take']))
        if int(fields['attacker'])==actor and fields['target_class'].startswith('monster_'):
            outgoing[fields['target_class']]+=damage
            if int(fields.get('shot','0')): hits.add(int(fields['shot']))
            kills+=int(fields['health_before'])>0 and int(fields['health_after'])<=0
        if int(fields['target'])==actor: incoming[fields['attacker_class']]+=damage
    return dict(root=str(root),seed=seed,counts=dict(counters),owners=dict(owners),guards=dict(guards),limits=dict(limits),
                yaw_error_mean=sum(yaw_errors)/len(yaw_errors) if yaw_errors else None,
                nearest_distance_mean=sum(distances)/len(distances) if distances else None,
                outgoing=dict(outgoing),incoming=dict(incoming),native_damage_shot_ids=len(hits),kills=kills,
                final_health=live[-1]['health'],goal_stop=(root/'goal-stop.json').exists(),
                trace_sha256=hashlib.sha256((root/'bot.jsonl').read_bytes()).hexdigest(),
                scope='First life after fixed release; minimum command yaw error to any clear observed enemy, horizontal only, without lead/pitch. Not hit accuracy. Low displacement is a stall proxy, not a proven wall collision.')

def batch(path,label,episode):
    p=pathlib.Path(path);report=read(p/'report.json')
    if not report.get('capture_complete'): raise ValueError('Incomplete cohort')
    result=[]
    for r in report['results']:
        if not r['capture_valid']: raise ValueError('Unverified capture')
        a=analyze(r['root']);a.update(label=label,episode=episode);result.append(a)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--experiment',required=True);ap.add_argument('--rules',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    results=[];experiment=read(pathlib.Path(args.experiment)/'report.json')
    for entry in experiment['leaderboard']:
        results+=batch(entry['artifacts'],entry['label'],entry['episode'])
    rules=read(pathlib.Path(args.rules)/'report.json')
    if rules['state']!='complete': raise ValueError('Rules pool incomplete')
    for entry in rules['records']:
        results+=batch(entry['artifacts'],'rules',entry['episode_id'])
    pathlib.Path(args.out).write_text(json.dumps(dict(version=1,results=results),indent=2),encoding='utf-8')
