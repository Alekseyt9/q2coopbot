"""Close a paired transition smoke with actor attribution and source hashes.

No neural numerical computation, no training acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def lines(path):
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:yield json.loads(line)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    capture=read(root/'report.json')
    assert capture['accepted'] and capture['source_unchanged'] and capture['native_source_unchanged']
    command=read(root/'command-receipt-audit.json')
    for path,digest in command['source_sha256'].items():assert sha(Path(path))==digest
    tamper=read(root/'tamper-verification.json')
    assert tamper['rejected'] and not tamper['output_created']
    result=[];hashes={}
    for role,name in enumerate(('PairLearner','PairLeader')):
        directory=root/f'dataset-role-{role}-v2'
        report=read(directory/'report.json')
        assert report['paired_confirmed'] and not report['paired_training_ready']
        assert report['observed_reset_confirmed'] and report['synchronous_confirmed']
        assert report['command_proof']['accepted'] and report['native_steps']==capture['completed_pairs']
        for path,digest in report['paired_source_sha256'].items():assert sha(Path(path))==digest
        outcome=list(lines(directory/'server_outcomes.jsonl'))
        step=list(lines(directory/'steps.jsonl'))
        assert len(outcome)==len(step)==report['steps']
        for s,o in zip(step,outcome):
            actor=s['observation']['identity']['actor']
            assert s['native_step']['actor']==actor==role+1
            assert s['server_execution']['matched']
            for e in o['events']:assert e['attacker']==actor or e['target']==actor
        counts={k:sum(o[k] for o in outcome) for k in ('monster_health_damage','monster_kills','received_health_damage','teammate_health_damage')}
        controlled=list(s for s in step if s['owner']=='provider')
        with_context=sum(s['observation'].get('navigation_context') is not None for s in controlled)
        result.append(dict(role=role,steps=len(step),provider_steps=len(controlled),
                           provider_steps_with_navigation=with_context,terminals=report['terminals'],
                           truncations=report['truncated'],effects=counts))
        for file in directory.iterdir():
            if file.is_file():hashes[str(file)]=sha(file)
    # Native logs show the rules peer killed the Soldier. Verify that none of
    # that damage/kill was silently credited to the learner actor.
    assert result[0]['effects']['monster_health_damage']==0 and result[0]['effects']['monster_kills']==0
    assert result[1]['effects']['monster_health_damage']==30 and result[1]['effects']['monster_kills']==1
    for name in ('report.json','command-receipt-audit.json','tamper-verification.json'):
        hashes[str(root/name)]=sha(root/name)
    closure=dict(version='coop_paired_transition_smoke_v1',results=result,source_sha256=hashes,
                 ppo_trainable=False,training_performed=False,
                 scope='Two-client command binding, selected-actor transition export and native effect attribution. One-site smoke; no terminal/reward/training eligibility or model superiority acceptance.')
    (root/'paired-export-verification.json').write_text(json.dumps(closure,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
