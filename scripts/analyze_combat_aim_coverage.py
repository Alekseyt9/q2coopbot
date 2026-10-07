"""Offline label coverage diagnostic; never emits training rows or game commands."""
import argparse,hashlib,json,math,pathlib

def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def stats(values):
    assert values
    return dict(rows=len(values),mean_degrees=sum(values)/len(values),over5=sum(v>5 for v in values),over20=sum(v>20 for v in values),over90=sum(v>90 for v in values))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bc-data',type=pathlib.Path,required=True);ap.add_argument('--capture',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
    meta=read(a.bc_data/'report.json');p=a.bc_data/'train.jsonl';assert sha(p)==meta['data_sha256']['train']
    teacher=[]
    for line in p.read_text().splitlines():
        r=json.loads(line)
        if r['mask'][1]:teacher.append(abs(r['targets'][2]*180))
    report=read(a.capture/'report.json');assert report['state']=='complete'
    receipts={str(p):sha(p),str(a.capture/'report.json'):sha(a.capture/'report.json')};learned=[]
    for binding in report['records']:
        assert binding['split']=='train' and binding['mode']=='learned'
        batch=pathlib.Path(binding['artifacts'])/'report.json';assert sha(batch)==binding['report_sha256'];receipts[str(batch)]=sha(batch)
        b=read(batch);assert b['capture_complete'] and b['provenance_valid']
        for result in b['results']:
            assert result['capture_valid'];trace=pathlib.Path(result['root'])/'bot.jsonl';receipts[str(trace)]=sha(trace);seen=set()
            for line in trace.read_text(encoding='utf-8-sig').splitlines():
                row=json.loads(line);c=row['combat_policy'];o=c['observation'];frame=o['identity']['frame']
                if o['health']<=0:break
                if c['selection']['owner']!='provider' or frame in seen:continue
                seen.add(frame);enemies=[e for e in o['enemies'] if e.get('clear_shot')]
                if not enemies:continue
                rel=min(enemies,key=lambda e:e['distance'])['relative'];goal=math.degrees(math.atan2(rel[1],rel[0]));yaw=o['view_angles'][1]*360/65536
                learned.append(abs((goal-yaw+180)%360-180))
    assert all(sha(p)==digest for p,digest in receipts.items()),'Input changed'
    output=dict(version='combat_aim_label_coverage_v1',teacher_applied_yaw_delta=stats(teacher),learned_train_nearest_visible_yaw_correction=stats(learned),source_sha256=receipts,scope='Teacher training labels versus unique first-life provider observations from learned TRAIN captures. Geometric horizontal correction to nearest clear observed enemy; no pitch, lead, recoil or target-optimality claim. These values are diagnostics, not counterfactual training labels or hit proof.')
    a.out.write_text(json.dumps(output,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in output.items() if k!='source_sha256'},indent=2))

if __name__=='__main__':main()
