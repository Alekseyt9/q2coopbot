"""Add exact observed-enemy H1 attribution to sealed auxiliary NPZ chunks."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();key=str(path);h=pins[key] if key in pins else sha(path)
        assert expected is None or expected==h,str(path);pins[key]=h;return path
    p=read(pin(a.protocol));assert p['version']=='combat_enemy_threat_predictor_experiment_v1'
    base=Path(p['base_dataset']);closed=read(pin(base/'complete.json'));ds=read(pin(base/'report.json',closed['report_sha256']))
    assert ds['version']=='combat_local_effect_prediction_dataset_v1'
    for path,h in ds['inputs'].items():pin(path,h)
    a.out.mkdir(parents=True);(a.out/'chunks').mkdir();chunks=[];totals={}
    for ci,chunk in enumerate(ds['chunks']):
        with np.load(pin(chunk['path'],chunk['sha256']),allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
        n=len(data['features']);incoming=np.zeros((n,8),dtype=np.float32);unknown=np.zeros(n,dtype=np.float32)
        if chunk['split']=='challenge':
            # With known total zero, nonnegative per-attacker health damage is
            # mathematically zero too; no unobserved future window is filled.
            assert data['known'][:,0].all() and (data['effects'][:,0,1]==0).all()
        else:
            folder=Path(chunk['case']['folder']);meta=read(pin(folder/'cuda/report.json'))
            sp=next((folder/'native').glob('replay-*/steps.jsonl'))
            steps=list(map(json.loads,pin(sp,meta['source_sha256'][str(sp.resolve())]).open(encoding='utf-8-sig')))
            byframe={s['observation']['identity']['frame']:s for s in steps if s['owner']=='provider' and s.get('native_step') and s['observation']['identity']['life']==1}
            ep=sp.with_name('server_outcomes.jsonl');effects={e['step']:e for e in map(json.loads,pin(ep,meta['source_sha256'][str(ep.resolve())]).open(encoding='utf-8-sig'))}
            for i,frame in enumerate(data['frame'].tolist()):
                assert data['known'][i,0]
                s=byframe[frame];o=s['observation'];effect=effects[s['index']];assert effect['available']
                enemies=sorted(o['enemies'],key=lambda e:e['distance'])[:8]
                slots={e['id']:(j,e['class']) for j,e in enumerate(enemies)}
                present=data['features'][i,73:169].reshape(8,12)[:,0]
                assert int(present.sum())==len(enemies)
                summed=0
                for e in effect['events']:
                    if e['target']!=o['identity']['actor']:continue
                    damage=min(max(e['health_before'],0),e['take']);assert damage>=0;summed+=damage
                    if e['attacker'] in slots and slots[e['attacker']][1]==e['attacker_class']:
                        incoming[i,slots[e['attacker']][0]]+=damage
                    else:unknown[i]+=damage
                assert summed==effect['received_health_damage']==data['effects'][i,0,1]
        assert np.array_equal(incoming.sum(1)+unknown,data['effects'][:,0,1])
        data.update(incoming_by_enemy=incoming,unknown_incoming=unknown)
        path=a.out/'chunks'/f'{ci:04d}-{chunk["split"]}.npz';np.savez_compressed(path,**data)
        chunks.append(dict(**{k:v for k,v in chunk.items() if k not in ('path','sha256')},path=str(path.resolve()),sha256=sha(path)))
        totals.setdefault(chunk['split'],dict(rows=0,visible_damage=0.,unknown_damage=0.))
        t=totals[chunk['split']];t['rows']+=n;t['visible_damage']+=float(incoming.sum());t['unknown_damage']+=float(unknown.sum())
        save(a.out/'progress.json',dict(stage='attributing',cases_done=ci+1,cases=len(ds['chunks']),totals=totals))
    for path,h in pins.items():assert sha(path)==h,path
    report=dict(state='complete',version='combat_enemy_threat_dataset_v1',protocol_sha256=sha(a.protocol),base_dataset_sha256=closed['report_sha256'],
        chunks=chunks,rows=ds['rows'],worlds=ds['worlds'],totals=totals,inputs=pins,all_h1_totals_exact=True,
        compressed_bytes=sum(Path(c['path']).stat().st_size for c in chunks),neural_execution=False,training_performed=False,new_games=False,
        actor_training=False,ordinary_training_eligible=False,promotion=False,
        scope='Observed enemy slots receive native H1targets only. Unobserved attackers preserved as separate residual. No IDs or native future state become model inputs. Existing whole-world splits retained.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    save(a.out/'progress.json',dict(stage='complete',totals=totals));print(dict(totals=totals,compressed_bytes=report['compressed_bytes']))


if __name__=='__main__':main()
