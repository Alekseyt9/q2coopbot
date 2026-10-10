"""Inspect recorded execution configs; never execute a neural model."""
import argparse, pathlib
from process_combat_architecture_pool import read, save, sha


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=pathlib.Path,required=True)
    a=ap.parse_args(); root=a.root.resolve(); protocol=read(root/'protocol.json')
    proof=read(root/'recovery/verified-members.json')
    assert proof['state']=='complete' and proof['protocol_sha256']==sha(root/'protocol.json')
    rows=[]; pairs={}
    for entry in protocol['evaluations']:
        plan=read(entry['plan'])
        if not plan.get('model_path'): continue
        original=read(plan['model_path'])
        for ti,task in enumerate(plan['tasks']):
            for seed in task['seeds']:
                folder=pathlib.Path(entry['root'])/f"case-{ti}-{task['modes'][0]}"/f's-{seed}'
                report=folder/'report.json'
                assert sha(report)==proof['members'][str(folder)]['report_sha256']
                result=read(report)['results'][0]
                candidates=list(folder.glob('provider-worker-*-episode-*.json')); assert len(candidates)==1
                config=candidates[0]; actual=read(config)
                assert sha(config)==result['provider_config_sha256'].lower()
                assert actual['sampling_seed']==seed+plan.get('policy_sampling_seed_offset',0)
                assert read(folder/'manifest.json').get('policy_sampling_seed_offset',0)==plan.get('policy_sampling_seed_offset',0)
                assert {k:v for k,v in actual.items() if k!='sampling_seed'}=={k:v for k,v in original.items() if k!='sampling_seed'}
                row=dict(model=entry['model'],label=entry['label'],family=task['episode']['id'],engine_seed=seed,
                    declared_sampling_seed=original['sampling_seed'],actual_sampling_seed=actual['sampling_seed'],
                    deterministic=actual['deterministic'],execution_config_sha256=sha(config),config=str(config))
                rows.append(row)
                if not actual['deterministic']:
                    pairs.setdefault((entry['model'],task['episode']['id'],seed),[]).append(row)
    duplicates=[dict(model=k[0],family=k[1],engine_seed=k[2],labels=[r['label'] for r in v])
        for k,v in pairs.items() if len(v)>1 and len({r['execution_config_sha256'] for r in v})==1]
    save(root/'sampling-config-audit.json',dict(state='complete',protocol_sha256=sha(root/'protocol.json'),
        member_proof_sha256=sha(root/'recovery/verified-members.json'),members=rows,
        repeated_stochastic_execution_configs=duplicates,
        scope='Recorded provider SHA checked against native receipt; all parameters except sampling_seed unchanged. Repeated execution configs are repeats, not independent policy RNG arms. No neural inference performed.'))
    print(f'configs={len(rows)}, repeated stochastic conditions={len(duplicates)}')


if __name__=='__main__': main()
