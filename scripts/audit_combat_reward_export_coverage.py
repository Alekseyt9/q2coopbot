"""TRAIN reward-label to CUDA export coverage; no neural execution or fitting."""
import argparse
import collections
import json
import math
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path);assert expected is None or digest==expected,str(path)
        pins[str(path)]=digest;return path
    closed=read(pin(a.corpus/'complete.json'));assert closed['state']=='complete' and not closed['training_performed']
    protocol=read(pin(a.corpus/'protocol.json',closed['protocol_sha256']))
    assert protocol['train_only'] and all(t['split']=='train' for t in protocol['assignments'])
    totals=collections.Counter();components=collections.Counter();cases=[];missing=[];max_error=0.
    for case in sorted((a.corpus/'processing-train').glob('case-*')):
        if not case.is_dir():continue
        meta=read(pin(case/'cuda/report.json'));assert meta['numerical_verification']=='cuda_verified_v1'
        files=list((case/'native').glob('replay-*/steps.jsonl'));assert len(files)==1
        step_path=files[0].resolve();reward_path=step_path.with_name('rewards.jsonl')
        pin(step_path,meta['source_sha256'][str(step_path)]);pin(reward_path,meta['source_sha256'][str(reward_path)])
        rollout=pin(case/'cuda/rollout.jsonl',meta['rollout_sha256'])
        steps={s['index']:s for s in map(json.loads,step_path.open(encoding='utf-8-sig'))}
        rewards={r['step']:r for r in map(json.loads,reward_path.open(encoding='utf-8-sig'))}
        rows={r['index']:r for r in map(json.loads,rollout.open(encoding='utf-8-sig'))}
        eligible={i:r for i,r in rewards.items() if r['available'] and i in steps and steps[i]['owner']=='provider' and steps[i]['observation']['identity']['life']==1}
        assert set(rows)<=set(eligible)
        c=collections.Counter()
        for i,r in eligible.items():
            error=abs(math.fsum(r['components'].values())-r['score']);max_error=max(max_error,error);assert error<1e-8
            totals['eligible_reward_rows']+=1;c['eligible_reward_rows']+=1
            totals['label_kill_reward']+=r['components'].get('monster_kill',0);totals['label_death_reward']+=r['components'].get('death',0)
            if i in rows:
                assert abs(rows[i]['reward']-r['score'])<1e-8
                totals['exported_reward_rows']+=1;c['exported_reward_rows']+=1
                totals['exported_kill_reward']+=r['components'].get('monster_kill',0);totals['exported_death_reward']+=r['components'].get('death',0)
                components.update(r['components'])
                c['kill_reward']+=r['components'].get('monster_kill',0);c['death_reward']+=r['components'].get('death',0)
            else:missing.append(dict(case=case.name,index=i,score=r['score'],components=r['components']))
        cases.append(dict(case=case.name,counts=dict(c),exported_terminal_rows=sum(r['terminal'] for r in rows.values()),
            exported_zero_bootstraps=sum(r['bootstrap_zero'] for r in rows.values())))
    assert len(cases)==closed['native_cases'] and not missing,'Available first-life provider rewards were dropped'
    assert totals['label_kill_reward']==totals['exported_kill_reward'] and totals['label_death_reward']==totals['exported_death_reward']
    for path,digest in pins.items():assert sha(Path(path))==digest
    a.out.mkdir(parents=True)
    report=dict(state='complete',version='combat_reward_export_coverage_v1',totals=dict(totals),component_sums=dict(components),cases=cases,
        missing_available_rewards=missing,component_score_max_error=max_error,inputs=pins,script_sha256=sha(__file__),
        neural_execution=False,training_performed=False,new_games=False,
        scope='Available first-life provider reward labels versus finalized CUDA rollout. This does not independently validate engine damage/death attribution, causal reward timing, missing unavailable labels, or advantage quality.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(cases=len(cases),totals=dict(totals),missing=len(missing),component_sums=dict(components)))


if __name__=='__main__':main()
