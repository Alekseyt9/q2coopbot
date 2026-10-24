"""Check whether observed per-enemy inputs can receive exact native H1 labels."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or expected==h,str(path);pins[str(path)]=h;return path
    closed=read(pin(a.dataset/'complete.json'));ds=read(pin(a.dataset/'report.json',closed['report_sha256']))
    assert ds['version']=='combat_local_effect_prediction_dataset_v1'
    totals={s:collections.Counter() for s in ('train','development')};classes={s:collections.defaultdict(collections.Counter) for s in totals}
    for chunk in ds['chunks']:
        split=chunk['split']
        if split=='challenge':continue
        assert chunk['case']['seed'] not in ds['worlds']['challenge']
        with np.load(pin(chunk['path'],chunk['sha256']),allow_pickle=False) as arrays:
            frames=arrays['frame'].tolist();targets=arrays['effects'][:,0,1].tolist();known=arrays['known'][:,0].tolist()
        folder=Path(chunk['case']['folder']);meta=read(pin(folder/'cuda/report.json'))
        sp=next((folder/'native').glob('replay-*/steps.jsonl'))
        steps=list(map(json.loads,pin(sp,meta['source_sha256'][str(sp.resolve())]).open(encoding='utf-8-sig')))
        byframe={s['observation']['identity']['frame']:s for s in steps if s['owner']=='provider' and s.get('native_step') and s['observation']['identity']['life']==1}
        ep=sp.with_name('server_outcomes.jsonl');effects={r['step']:r for r in map(json.loads,pin(ep,meta['source_sha256'][str(ep.resolve())]).open(encoding='utf-8-sig'))}
        for frame,target,valid in zip(frames,targets,known):
            assert valid
            s=byframe[frame];o=s['observation'];effect=effects[s['index']];assert effect['available'] and effect['received_health_damage']==target
            visible={e['id']:e for e in o['enemies']};remembered={e['id']:e for e in (o.get('remembered_threats') or [])}
            counter=totals[split];counter['rows']+=1;counter['visible_enemy_slots']+=len(visible)
            for e in visible.values():
                classes[split][e['class']]['visible_slots']+=1
                classes[split][e['class']]['animation_known_slots']+=e.get('observed_animation_frame') is not None
            summed=0
            for e in effect['events']:
                if e['target']!=o['identity']['actor']:continue
                damage=min(max(e['health_before'],0),e['take']);assert damage>=0
                summed+=damage
                if damage==0:continue
                attacker=e['attacker'];kind='unknown'
                if attacker==o['identity']['actor']:kind='self'
                elif attacker in visible and visible[attacker]['class']==e['attacker_class']:kind='visible'
                elif attacker in remembered and remembered[attacker]['class']==e['attacker_class']:kind='remembered'
                counter[kind+'_events']+=1;counter[kind+'_health_damage']+=damage
                classes[split][e['attacker_class']][kind+'_events']+=1
                classes[split][e['attacker_class']][kind+'_health_damage']+=damage
                if kind=='visible' and visible[attacker].get('observed_animation_frame') is not None:
                    counter['visible_damage_with_animation']+=damage
            assert summed==target,(chunk['case'],frame,summed,target)
            counter['received_health_damage']+=target;counter['positive_rows']+=target>0
    for path,h in pins.items():assert sha(path)==h,path
    summary={s:dict(counts=dict(c),visible_damage_fraction=c['visible_health_damage']/c['received_health_damage'] if c['received_health_damage'] else None,
                    visible_damage_animation_fraction=c['visible_damage_with_animation']/c['visible_health_damage'] if c['visible_health_damage'] else None,
                    classes={k:dict(v) for k,v in classes[s].items()}) for s,c in totals.items()}
    a.out.mkdir(parents=True);report=dict(state='complete',version='combat_incoming_attribution_audit_v1',summary=summary,inputs=pins,
        all_h1_aggregates_reconstructed=True,neural_execution=False,training_performed=False,new_games=False,promotion=False,
        scope='Feasibility of separate native H1 damage targets for currently observed enemy slots. Repeated branch rows are correlated, not independent attacks. Attacker IDs/classes only join offline labels; no hidden attacker features become NN inputs. Unobserved attackers remain separate, never negative/zero-filled visible labels.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(json.dumps(summary))


if __name__=='__main__':main()
