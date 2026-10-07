"""Join native use receipt to first-life UDP-visible equipped weapon."""
import argparse,json,pathlib,re

def audit(batch):
    report=json.loads((batch/'report.json').read_text(encoding='utf-8-sig'))
    if not report['provenance_valid'] or not report['capture_complete']:raise ValueError('Incomplete native capture')
    results=[]
    for worker in report['results']:
        if not worker['capture_valid'] or not worker['dispatch_valid']:raise ValueError('Unverified execution')
        root=pathlib.Path(worker['root'])
        native=(root/'server.log').read_text(encoding='utf-8-sig')
        requests=[int(frame) for frame in re.findall(r'sv_test_client_cmd frame=(\d+) seq=\d+ state=3 command=use Blaster\b',native)]
        if not requests:raise ValueError('Native Blaster request missing')
        first=worker['first_life'];start=first['start_frame'];end=first['end_frame']
        requests=[frame for frame in requests if start<=frame<end]
        if not requests:raise ValueError('No first-life request')
        rows=[json.loads(s) for s in (root/'bot.jsonl').read_text(encoding='utf-8-sig').splitlines()]
        observations=[r for r in rows if requests[0]<=r.get('frame',0)<=end and r.get('health',0)>0 and r.get('weapon')=='Blaster']
        if not observations:raise ValueError('No first-life equipped Blaster observation')
        switched=observations[0]
        if switched['map']!=first['map'] or switched['spawncount']!=first['spawncount']:raise ValueError('Identity mismatch')
        results.append(dict(seed=worker['seed'],native_request_frame=requests[0],observed_equipped_frame=switched['frame'],health=switched['health'],delay_frames=switched['frame']-requests[0]))
    return dict(version='combat_weapon_switch_audit_v1',accepted=True,results=results,scope='Native request plus UDP-visible equipped Blaster in first life. Diagnostic release-fire probe, not learned weapon quality or full weapon set acceptance.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--batch',required=True,type=pathlib.Path);ap.add_argument('--out',required=True,type=pathlib.Path);a=ap.parse_args()
    result=audit(a.batch)
    with a.out.open('x',encoding='utf-8') as output:json.dump(result,output,indent=2)
    print(json.dumps(result,indent=2))
