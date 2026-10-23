"""Replay an unfinished handoff with an explicit objective fork, diagnostic only.

Native observed actions/features remain unchanged; no new game, no fitting.
The critic and all behavior numerical checks execute exclusively on CUDA.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from evaluate_registered_combat_models import read, save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('case', 'capture', 'model', 'exporter', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args(); assert not a.out.exists()
    case, capture, out = (p.resolve() for p in (a.case, a.capture, a.out))
    repo = Path(__file__).resolve().parents[1]
    metadata = read(case/'native/report.json'); result = read(capture/'report.json')['results'][0]
    assert not result['goal_stop'] and not result['death_stop']
    worker = Path(result['root']); old = next((case/'native').glob('replay-*/steps.jsonl')).parent
    config = read(old/'reward-config.json'); assert not config.get('handoff_bootstrap')
    config['handoff_bootstrap'] = 'nonterminal_v1'
    out.mkdir(parents=True); save(out/'reward-config.json', config)
    command = [str(a.exporter.resolve()), '--trace', str(worker/'bot.jsonl'), '--server-log', str(worker/'server.log'),
        '--reset-expectation', str(worker/'reset-expectation.json'), '--reward-config', str(out/'reward-config.json'),
        '--out', str(out/'replay'), '--worker', 'worker-0', '--episode', 'seed-'+str(result['seed']),
        '--end-reason', 'game_frame_limit', '--client-name', 'SoloRetreatBot', '--require-execution', '--synchronous']
    with (out/'replay.log').open('w', encoding='utf-8') as log:
        r = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
    assert r.returncode == 0, str(out/'replay.log')
    records = lambda p: [json.loads(line) for line in p.open(encoding='utf-8-sig')]
    old_steps = records(old/'steps.jsonl'); steps = records(out/'replay/steps.jsonl')
    assert steps == old_steps, 'Objective-only replay changed observed action/state records'
    rewards = {r['step']: r for r in records(out/'replay/rewards.jsonl')}
    by_index = {s['index']: s for s in steps}
    original = records(case/'native/native-rollout.jsonl'); rows = copy.deepcopy(original)
    changed = []
    for row in rows:
        step = by_index[row['index']]; reward = rewards[row['index']]; assert reward['available']
        row['reward'] = reward['score']
        row['bootstrap_zero'] = bool(step['terminal'] or step.get('goal_handoff_verified', False))
        if step.get('reason') == 'control_handoff':
            assert step['next_observation']['health'] > 0 and not row['bootstrap_zero']
            changed.append(dict(index=row['index'], frame=row['frame'], next_frame=row['next_frame']))
    assert len(changed) == 1
    for before, after in zip(original, rows):
        assert {k:v for k,v in before.items() if k not in ('reward', 'bootstrap_zero')} == {k:v for k,v in after.items() if k not in ('reward', 'bootstrap_zero')}
    native = out/'native'; native.mkdir()
    with (native/'native-rollout.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows: stream.write(json.dumps(row, allow_nan=False)+'\n')
    # Sequence is immutable and shared by path only via an ordinary small copy.
    import shutil
    shutil.copy2(case/'native/sequence.jsonl', native/'sequence.jsonl')
    sources = dict(metadata['source_sha256'])
    for p in (case/'native/report.json', a.exporter.resolve(), out/'reward-config.json',
              out/'replay/steps.jsonl', out/'replay/rewards.jsonl', out/'replay/report.json', Path(__file__).resolve()):
        sources[str(p.resolve())] = sha(p)
    fresh = copy.deepcopy(metadata)
    fresh.update(version='combat_ppo_native_cuda_pending_v1', source_sha256=sources,
        native_rollout_sha256=sha(native/'native-rollout.jsonl'), sequence_sha256=sha(native/'sequence.jsonl'),
        reward_config_sha256=sha(out/'reward-config.json'), diagnostic_only=True, ppo_training_ready=False,
        handoff_bootstrap='nonterminal_v1', scope='Diagnostic objective-only replay; observed actions and features unchanged, not a training corpus.')
    save(native/'report.json', fresh)
    with (out/'cuda.log').open('w', encoding='utf-8') as log:
        r = subprocess.run([sys.executable, str(repo/'scripts/finalize_combat_cuda_rollout.py'),
            '--model', str(a.model.resolve()), '--data', str(native), '--out', str(out/'cuda')], cwd=repo, stdout=log, stderr=subprocess.STDOUT)
    assert r.returncode == 0, str(out/'cuda.log')
    after = records(out/'cuda/rollout.jsonl'); target = next(r for r in after if r['index'] == changed[0]['index'])
    report = dict(state='complete', version='combat_nonterminal_handoff_diagnostic_v1', new_games=False, training_performed=False,
        device='cuda', diagnostic_only=True, seed=result['seed'], rows=len(rows), observed_states_actions_exact=True,
        unfinished_handoff=changed[0], previous_next_value=next(r['next_value'] for r in records(case/'cuda/rollout.jsonl') if r['index'] == target['index']),
        next_value=target['next_value'], bootstrap_zero=target['bootstrap_zero'], cuda_proof_sha256=sha(out/'cuda/cuda-verification.json'),
        reward_config_sha256=sha(out/'reward-config.json'), promotion=False, ordinary_training_eligible=False)
    assert not report['bootstrap_zero']
    save(out/'report.json', report); save(out/'complete.json', dict(state='complete', report_sha256=sha(out/'report.json')))
    print(report)


if __name__ == '__main__': main()
