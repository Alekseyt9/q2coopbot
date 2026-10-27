"""Bind teacher root actions to actual native MG fire, without fitting or relabelling."""
import argparse
import collections
import json
from pathlib import Path

from combat_native_teacher_registry import load_index
from process_combat_architecture_pool import read,save,sha
from report_combat_machinegun_hits import scan
from report_registered_machinegun_outcomes import outcome


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();repo=Path(__file__).resolve().parents[1];out=a.out.resolve()
    assert not out.exists();records=load_index(repo,a.index);out.mkdir(parents=True)
    pins={str(a.index.resolve()):sha(a.index),str(Path(__file__).resolve()):sha(__file__)}
    for name in ('report_combat_machinegun_hits.py','report_registered_machinegun_outcomes.py','native_projectile_window.py'):
        p=repo/'scripts'/name;pins[str(p)]=sha(p)
    for corpus in read(a.index)['corpora']:
        manifest=(repo/corpus['manifest']).resolve();pins[str(manifest)]=sha(manifest)
        for entry in read(manifest)['entries']:pins[entry['path']]=entry['sha256']
    save(out/'protocol.json',dict(version='combat_teacher_actual_root_fire_v1',
        scope='Descriptive native root-window audit only. All original contexts retained. No fitting,selection,reward or teacher-label changes.',
        absence='Zero root shots is confirmed only if strict MG fire telemetry exists elsewhere in that branch log. A held attack button does not imply a shot;no shot alone does not prove cooldown.',
        window='Native ordered begin/end command window matched by spawncount,begin_frame,actor,sequence;never lookup by game frame alone.',
        inputs_sha256=pins,training_performed=False,promotion=False))
    cases=[]
    for record in records:
        branches={}
        for mode in ('choice0','choice1'):
            binding=record['ownership_admission']['evidence'][mode]
            steps_path=Path(binding['steps_path']);native_path=Path(binding['native_report'])
            assert sha(steps_path)==binding['steps_sha256'] and sha(native_path)==binding['native_report_sha256']
            native=read(native_path)
            logs=[Path(p) for p in native['source_sha256'] if p.endswith('server.log')]
            assert len(logs)==1;log=logs[0]
            assert sha(log)==native['source_sha256'][str(log)]
            with steps_path.open(encoding='utf-8-sig') as stream:
                roots=[r for r in map(json.loads,stream) if r['observation']['identity']['frame']==record['frame']]
            assert len(roots)==1;root=roots[0];n=root['native_step'];execution=root['server_execution']
            assert execution['matched'] and execution['window_exclusive'] and execution['recovery_commands']==0
            assert root['owner']=='provider' and n['begin_frame']==record['frame']
            assert root['sent_command']==record['outcomes'][mode]['native']['root_sent']
            window=(n['spawncount'],n['begin_frame'],n['actor'],n['sequence'])
            with log.open(encoding='utf-8-sig') as stream:all_shots=scan(stream,{window})
            shots=[s for s in all_shots if s['eligible']]
            damage=0
            for shot in shots:
                if outcome(shot)=='live_monster_damage':
                    d=shot['damage'];damage+=max(0,int(d['health_before']))-max(0,int(d['health_after']))
            status='fired' if shots else 'no_root_fire' if all_shots else 'unconfirmed_mg_telemetry'
            branches[mode]=dict(status=status,attack_sample=record['outcomes'][mode]['native']['root_attack'],
                attack_sent=bool(root['sent_command']['Buttons']&1),shots=len(shots),root_live_monster_hp=damage,
                observed_weapon=root['observation']['weapon'],observed_gun_frame=root['observation']['gun_frame'],
                observed_ammo=root['observation']['ammo'],requested_weapon=root['action'].get('weapon'),
                native_window=window,full_log_mg_shots=len(all_shots),
                outcomes=[outcome(s) for s in shots],
                native_firing_gunframes=[int(s['fields']['gunframe']) for s in shots],
                server_log=str(log),server_log_sha256=sha(log))
            for p in (steps_path,native_path,log):pins[str(p)]=sha(p)
        b0,b1=branches['choice0'],branches['choice1']
        unknown=any(b['status']=='unconfirmed_mg_telemetry' for b in (b0,b1))
        fired=int(b0['shots']>0)+int(b1['shots']>0)
        category='unknown' if unknown else ('neither_fired','one_fired','both_fired')[fired]
        delta=b1['root_live_monster_hp']-b0['root_live_monster_hp']
        immediate_preference=None if delta==0 else int(delta>0)
        preference=record['target']['preference']
        cases.append(dict(id=record['id'],map=record['map'],seed=record['seed'],frame=record['frame'],
            teacher_preference=preference,teacher_eligible=record['target']['eligible_for_teacher_supervision'],
            category=category,root_hp_delta=delta,immediate_preference=immediate_preference,
            long_immediate_disagree=(preference is not None and immediate_preference is not None and preference!=immediate_preference),
            branches=branches))
        save(out/'progress.json',dict(stage='native_root_fire',completed_records=len(cases),records=len(records)))
    def summarize(rows):
        return dict(records=len(rows),categories=dict(collections.Counter(r['category'] for r in rows)),
            nonzero_root_damage_pairs=sum(r['root_hp_delta']!=0 for r in rows),
            long_immediate_disagreements=sum(r['long_immediate_disagree'] for r in rows),
            attack_without_root_fire_branches=sum(b['attack_sent'] and b['status']=='no_root_fire' for r in rows for b in r['branches'].values()),
            roots_with_any_attack_sent=sum(any(b['attack_sent'] for b in r['branches'].values()) for r in rows))
    for p,digest in pins.items():assert sha(p)==digest,p
    save(out/'report.json',dict(state='complete',all=summarize(cases),labelled=summarize([r for r in cases if r['teacher_preference'] is not None]),
        cases=cases,inputs_sha256=pins,training_performed=False,neural_execution=False,promotion=False,
        interpretation='Actual MG fire and immediate living-monster damage in the intervened command window. Full-return labels remain valid controlled finite-horizon outcomes even without an immediate shot;they are not isolated shot-aim labels. Other weapons outside this firing denominator.'))
    save(out/'progress.json',dict(stage='complete'));print(json.dumps(dict(all=summarize(cases),labelled=summarize([r for r in cases if r['teacher_preference'] is not None]))))


if __name__=='__main__':main()
