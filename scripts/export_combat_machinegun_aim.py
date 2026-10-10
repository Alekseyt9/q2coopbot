"""Native recoil/muzzle are offline labels; client feature bytes remain unchanged."""
import argparse,gzip,json,math,pathlib,collections
from process_combat_architecture_pool import read,save,sha
from report_combat_machinegun_hits import scan,vector
from report_combat_selected_target_aim import wrap

def labels(shot,step):
    o=step['observation'];f=shot['fields'];start=vector(f['start']);kick=vector(f['recoil'])
    result=[]
    if step['owner']!='provider' or any(i['component'] in ('aim','pitch') for i in step.get('interventions',[])):return result
    for slot,e in enumerate(sorted(o['enemies'],key=lambda e:e['distance'])[:8]):
        solid=e.get('observed_solid',0);bottom=-((solid>>5)&31)*8;top=((solid>>10)&63)*8-32
        if not e.get('clear_shot') or solid in (0,31) or solid&31==0 or top<=bottom:continue
        height=(top+bottom)/2 if top-bottom<16 else min(max(22,bottom+8),top-8)
        p=[o['position'][i]+e['relative'][i]-start[i] for i in range(3)];p[2]+=height
        if math.hypot(*p[:2])<1e-9:continue
        yaw=wrap(math.degrees(math.atan2(p[1],p[0]))-kick[1]-o['view_angles'][1]*360/65536)
        pitch=wrap(-math.degrees(math.atan2(p[2],math.hypot(*p[:2])))-kick[0]-o['view_angles'][0]*360/65536)
        result.append(dict(slot=slot+1,entity=e['id'],track=e.get('observed_track',0),yaw_delta_degrees=yaw,pitch_delta_degrees=pitch))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--capture',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
    root=a.capture.resolve();out=a.out.resolve();assert not out.exists()
    protocol=read(root/'protocol.json');proof=read(root/'recovery/verified-members.json');assert proof['state']=='complete' and proof['protocol_sha256']==sha(root/'protocol.json')
    spec=read(root/'corpus-spec.json');meta=read(root/'client-queries/report.json');joined={};sources=[]
    for member in spec['members']:
        path=pathlib.Path(member['report']);assert proof['members'][str(path.parent)]['report_sha256']==member['report_sha256']==sha(path)
        steps=pathlib.Path(member['steps']);log=pathlib.Path(member['server']);assert sha(steps)==member['steps_sha256'] and sha(log)==member['server_sha256']
        by_window={}
        with steps.open(encoding='utf-8-sig') as stream:
            for line in stream:
                s=json.loads(line);o=s['observation'];n=s.get('native_step');ex=s.get('server_execution',{})
                if n and ex.get('matched') and ex.get('window_exclusive') and ex.get('recovery_commands')==0 and o['identity']['life']==1 and o['identity']['frame']>100 and o['health']>0:
                    key=(n['spawncount'],n['begin_frame'],n['actor'],n['sequence']);assert key not in by_window;by_window[key]=s
        with log.open(encoding='utf-8-sig') as stream:shots=scan(stream,set(by_window))
        for shot in shots:
            if not shot['eligible']:continue
            step=by_window[shot['window']];key=(member['seed'],step['index']);assert key not in joined,'More than one MG shot in a command'
            joined[key]=labels(shot,step)
        sources.append(dict(seed=member['seed'],split=member['split'],steps_sha256=sha(steps),server_sha256=sha(log)))
    out.mkdir();counts={};hashes={};seeds={}
    for split in ('train','validation'):
        source=root/'client-queries'/f'{split}.jsonl.gz';assert sha(source)==meta['data_sha256'][split]
        total=mg=blaster=0;seedset=set()
        with gzip.open(source,'rt',encoding='utf-8') as src,gzip.open(out/f'{split}.jsonl.gz','wt',encoding='utf-8') as dest:
            for line in src:
                row=json.loads(line);seedset.add(row['seed']);features=list(row['features']);assert len(features)==854
                native=joined.get((row['seed'],row['step']))
                # MG labels have their own native proof; postmove query masking
                # must not silently discard a shot in the final living command.
                queries=native if native is not None else [q for q in row['target_queries'] or [] if q['lead_known'] and q['recoil_known']] if row['mask'][1] else []
                row['supervised_aim_queries']=queries;row['aim_label_weapon']='machinegun' if native is not None else 'blaster'
                assert row['features']==features
                mg+=len(queries) if native is not None else 0;blaster+=len(queries) if native is None else 0;total+=1
                dest.write(json.dumps(row,separators=(',',':'))+'\n')
        assert mg>0 and blaster>0;counts[split]=dict(rows=total,machinegun_pairs=mg,blaster_pairs=blaster);seeds[split]=seedset;hashes[split]=sha(out/f'{split}.jsonl.gz')
    assert not seeds['train']&seeds['validation']
    save(out/'report.json',dict(version='native_conditional_aim_labels_v1',state='complete',counts=counts,data_sha256=hashes,sources=sources,protocol_sha256=sha(root/'protocol.json'),client_queries_report_sha256=sha(root/'client-queries/report.json'),features_preserved=True,
        scope='Client pre-command features unchanged. MG labels use actual native muzzle and fresh recoil only offline, against pre-command observed bbox points for each available target. Conditional on executed shot/movement, not expert target or tactical labels; camera kick is not equated with fresh recoil. Blaster postmove queries retained. Training and development validation disjoint; final test deferred.'))
    print(json.dumps(counts),flush=True)

if __name__=='__main__':main()
