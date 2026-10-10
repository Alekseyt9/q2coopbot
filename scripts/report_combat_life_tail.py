"""Count exported rows masked after first life in sealed native evaluations."""
import argparse
import collections
import json
import pathlib
from process_combat_architecture_pool import read, save, sha
from run_combat_spatial_ppo import wait_process


def report(root):
    protocol, proof = read(root/'protocol.json'), read(root/'recovery/verified-members.json')
    assert proof['state'] == 'complete' and proof['source_unchanged']
    assert proof['protocol_sha256'] == sha(root/'protocol.json')
    groups, sources = {}, []
    for entry in protocol['evaluations']:
        plan = read(entry['plan'])
        assert sha(entry['plan']) == entry['plan_sha256']
        rows = []
        for index, task in enumerate(plan['tasks']):
            for seed in task['seeds']:
                member = pathlib.Path(entry['root'])/f"case-{index}-{task['modes'][0]}"/f's-{seed}'
                path = member/'report.json'
                digest = sha(path)
                assert digest == proof['members'][str(member)]['report_sha256']
                capture = read(path)
                assert capture['capture_complete'] and capture['provenance_valid'] and len(capture['results']) == 1
                result = capture['results'][0]
                assert result['seed'] == seed and result['capture_valid'] and result['dispatch_valid']
                data = result['dataset']
                masks = data.get('reward_masks', {})
                after = masks.get('after_first_life', 0)
                assert 0 <= after <= data['steps'] and 0 <= data['reward_steps'] <= data['steps']
                assert sum(masks.values())+data['reward_steps'] == data['steps']
                rows.append(dict(seed=seed, family=task['episode']['id'], exported_rows=data['steps'],
                    rewarded_rows=data['reward_steps'], masked_after_first_life_rows=after,
                    all_reward_masks=masks, first_life_end_reason=result['first_life']['end_reason'],
                    capture_game_frames=result['actual_game_frames']))
                assert sha(path) == digest
                sources.append(dict(path=str(path), sha256=digest))
        assert len(rows) == protocol['episodes_per_model']
        totals = collections.Counter()
        for row in rows:
            for field in ('exported_rows', 'rewarded_rows', 'masked_after_first_life_rows', 'capture_game_frames'):
                totals[field] += row[field]
        groups[entry['model']+'-'+entry['label']] = dict(episodes=rows, totals=dict(totals),
            deaths=sum(r['first_life_end_reason'] == 'first_observed_death' for r in rows),
            after_first_life_exported_fraction=totals['masked_after_first_life_rows']/totals['exported_rows'] if totals['exported_rows'] else None)
    save(root/'first-life-tail.json', dict(version='combat_first_life_tail_v1', state='complete', groups=groups,
        protocol_sha256=sha(root/'protocol.json'), member_proof_sha256=sha(root/'recovery/verified-members.json'),
        sources=sources, scope='Export metadata from complete sealed captures. Masked rows after first life include dead intervals and later lives; excluded from current first-life rewards. Row fraction is not wall-clock savings or a count of all server frames. Preparation, startup and exporter overhead remain. No gameplay or termination changes; preserving the death transition and native death proof is required before introducing early stop.'))
    return {name: dict(totals=group['totals'], deaths=group['deaths'],
        after_first_life_exported_fraction=group['after_first_life_exported_fraction']) for name, group in groups.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.wait_pid:
        wait_process(args.wait_pid, root/'first-life-tail-progress.json', stage='waiting_for_sealed_evaluation')
    try:
        result = report(root)
        save(root/'first-life-tail-progress.json', dict(state='complete', report=str(root/'first-life-tail.json')))
        print(json.dumps(result), flush=True)
    except Exception as error:
        save(root/'first-life-tail-progress.json', dict(stage='failed', error=str(error)))
        raise
