"""CUDA geometry audit of native shots; server facts never enter model inputs."""
import argparse,json,pathlib,re
from process_combat_architecture_pool import read,sha,save
from train_combat_bc import torch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
    assert torch.cuda.is_available();protocol=read(a.root/'protocol.json');proof=read(a.root/'recovery/verified-members.json')
    assert proof['state']=='complete' and proof['protocol_sha256']==sha(a.root/'protocol.json')
    rows=[];sources=[]
    pattern=re.compile(r'^sv_test_projectile spawncount=(-?\d+) server_frame=(\d+) g_test_projectile version=1 event=spawn map=\w+ frame=\d+ shot=\d+ entity=\d+ attacker=(\d+) mod=1 x=([^ ]+) y=([^ ]+) z=([^ ]+) vx=([^ ]+) vy=([^ ]+) vz=([^ ]+)')
    for entry in protocol['evaluations']:
        for path in pathlib.Path(entry['root']).glob('case-*/s-*/report.json'):
            report=read(path);assert report['capture_complete'] and report['provenance_valid'];root=pathlib.Path(report['results'][0]['root'])
            assert proof['members'][str(path.parent)]['report_sha256']==sha(path)
            steps=root/'dataset/steps.jsonl';log=root/'server.log';indexed={}
            with steps.open(encoding='utf-8-sig') as f:
                for line in f:
                    s=json.loads(line);n=s.get('native_step');o=s['observation']
                    if n and o['identity']['life']==1 and o['identity']['frame']>100 and o['health']>0 and s.get('server_execution',{}).get('matched'):
                        indexed[(n['spawncount'],n['begin_frame'],n['actor'])]=s
            with log.open(encoding='utf-8-sig') as f:
                for line in f:
                    m=pattern.match(line.strip())
                    if not m:continue
                    s=indexed.get(tuple(map(int,m.groups()[:3])))
                    if not s:continue
                    o=s['observation'];next_o=s['next_observation']
                    rows.append(dict(group=entry['model']+'-'+entry['label'],position=o['position'],angles=o['view_angles'][:2],next_angles=next_o['view_angles'][:2],ducked=o['ducked'],start=list(map(float,m.groups()[3:6])),velocity=list(map(float,m.groups()[6:9]))))
            sources.append(dict(steps=str(steps),steps_sha256=sha(steps),server=str(log),server_sha256=sha(log)))
    assert rows
    def tensor(key):return torch.tensor([r[key] for r in rows],device='cuda')
    def forward(angles):
        pitch,yaw=(angles.float()* (2*torch.pi/65536)).unbind(-1)
        return torch.stack((pitch.cos()*yaw.cos(),pitch.cos()*yaw.sin(),-pitch.sin()),-1)
    d=forward(tensor('angles'));next_d=forward(tensor('next_angles'));velocity=tensor('velocity');start=tensor('start');position=tensor('position')
    height=torch.where(tensor('ducked').bool(),-2.,22.)-8
    predicted=position+24*(velocity/1000);predicted[:,2]+=height
    direction_error=(velocity/1000-d).norm(dim=-1);next_error=(velocity/1000-next_d).norm(dim=-1);muzzle_error=(start-predicted).norm(dim=-1)
    bad=(direction_error>=.001).nonzero().flatten().tolist()
    report=dict(mismatch_count=len(bad),state='passed' if bool(((velocity.norm(dim=-1)-1000).abs()<.01).all() & (muzzle_error<.2).all()) else 'mismatch',device='cuda',shots=len(rows),direction_vector_max_error=float(direction_error.max()),muzzle_position_max_error=float(muzzle_error.max()),next_observation_direction_mean_error=float(next_error.mean()),current_observation_direction_mean_error=float(direction_error.mean()),sources=sources,protocol_sha256=sha(a.root/'protocol.json'),scope='Native blaster mod1, first-life shots joined by exact dispatch frame/actor/spawncount. Center-hand muzzle24native_direction +viewheight-8 validated against observed origin. Native velocity is diagnostic only. Current observation angle usually matches firing direction; mismatch_count preserves exceptions, no universal phase claim. No Go model inference. Distinguish native firing angle from newly applied command; no immediate shot attribution assumed.')
    save(a.out,report);print(json.dumps({k:v for k,v in report.items() if k!='sources'}),flush=True)
    assert report['state']=='passed'

if __name__=='__main__':main()
