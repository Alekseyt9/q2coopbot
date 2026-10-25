"""Native-verified movement/stop/context balance before any neural fitting."""
import argparse
import math
from collections import Counter
from pathlib import Path
import numpy as np
from audit_combat_contact_loss_probe import load_steps
from audit_combat_route_motion import command_yaw
from audit_combat_movement_coverage import route_active
from evaluate_registered_combat_models import read, save, sha


def summarize(rows):
    moving = [r for r in rows if r['magnitude'] > 0]
    zero = [r for r in rows if r['magnitude'] == 0]
    yaw = sorted(abs(r['yaw']) for r in moving)
    displacements = [r['displacement'] for r in zero if r['displacement'] is not None]
    return dict(rows=len(rows), moving_rows=len(moving), zero_command_rows=len(zero),
                moving_fraction=len(moving)/len(rows) if rows else None,
                always_zero_local_label_mae=sum(r['zero_error'] for r in rows)/len(rows) if rows else None,
                moving_command_bins=dict(Counter('slow' if r['magnitude'] < 100 else 'medium' if r['magnitude'] < 250 else 'fast' for r in moving)),
                context_counts=dict(Counter(r['context'] for r in rows)),
                moving_context_counts=dict(Counter(r['context'] for r in moving)),
                applied_pose_counts=dict(Counter(r['pose'] for r in rows)),
                moving_mean_abs_yaw_degrees=sum(yaw)/len(yaw) if yaw else None,
                moving_p90_abs_yaw_degrees=yaw[round((len(yaw)-1)*.9)] if yaw else None,
                zero_command_native_next_rows=len(displacements),
                zero_command_native_displacement_lt1=sum(d < 1 for d in displacements))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--demonstrations', required=True, type=Path, nargs='+')
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    assert not a.out.exists()
    pins = {str(Path(__file__).resolve()): sha(__file__)}
    corpora = []
    for path in a.demonstrations:
        report = read(path)
        assert report['version'] == 'combat_movement_demonstrations_v1' and report['state'] == 'passed'
        assert report['ppo_eligible'] is False
        pins[str(path.resolve())] = sha(path)
        for p, h in report['inputs'].items():
            assert sha(p) == h
        rows, episodes = [], []
        worlds = {'train': set(), 'diagnostic': set()}
        for e in report['episodes']:
            assert e['split'] in worlds
            key = (e['map'], e['seed'])
            assert key not in worlds[e['split']]
            worlds[e['split']].add(key)
            pack = Path(e['pack'])
            assert sha(pack) == e['pack_sha256']
            pins[str(pack.resolve())] = sha(pack)
            native_path, obs_path = pack.parent/'native/steps.jsonl', pack.parent/'observations.jsonl'
            pins[str(native_path.resolve())], pins[str(obs_path.resolve())] = sha(native_path), sha(obs_path)
            steps = load_steps(native_path)
            by_frame = {s['observation']['identity']['frame']: s for s in steps}
            obs = load_steps(obs_path)
            arrays = np.load(pack, allow_pickle=False)
            assert arrays['features'].shape == (len(obs), 1554)
            indices = arrays['indices']
            assert len(indices) == e['rows'] and len(indices) == len(set(map(int, indices)))
            assert arrays['movement'].shape == (len(indices), 2)
            episode_rows = []
            for j, index in enumerate(indices):
                assert 0 <= index < len(obs)
                o = obs[index]
                s = by_frame[o['identity']['frame']]
                c, applied = s['sent_command'], s['applied_action']
                assert s['observation'] == o and s['owner'] == 'rules'
                assert s['native_step'] and s['server_execution']['matched'] and s['server_execution']['window_exclusive']
                assert np.allclose(arrays['movement'][j], [c['Forward']/400, c['Side']/400], rtol=0, atol=1e-7)
                assert abs(float(arrays['applied_yaw_degrees'][j])-applied['yaw_delta_degrees']) < 1e-5
                n = s['next_observation']
                if n is not None:
                    assert command_yaw(o, applied) == n['view_angles'][1]
                x = arrays['features'][index]
                context = 'route' if route_active(x) else ('visible' if any(x[73+12*k] for k in range(8)) else 'no_visible_no_route')
                row = dict(split=e['split'], magnitude=math.hypot(c['Forward'], c['Side']), context=context,
                           yaw=applied['yaw_delta_degrees'], pose=applied['vertical'],
                           zero_error=(abs(c['Forward'])+abs(c['Side']))/800,
                           displacement=math.hypot(*(n['position'][k]-o['position'][k] for k in (0, 1))) if n is not None else None)
                episode_rows.append(row)
            rows.extend(episode_rows)
            episodes.append(dict(map=e['map'], seed=e['seed'], split=e['split'], **summarize(episode_rows)))
        assert not worlds['train'] & worlds['diagnostic']
        corpora.append(dict(source=str(path.resolve()), worlds={s: len(w) for s, w in worlds.items()},
                            totals={s: summarize([r for r in rows if r['split'] == s]) for s in worlds}, episodes=episodes))
    assert all(sha(p) == h for p, h in pins.items())
    save(a.out, dict(version='combat_movement_demonstration_balance_v1', state='passed', corpora=corpora, inputs=pins,
                     training_performed=False, neural_execution=False, promotion=False,
                     scope='Native executed successful-rule demonstration coverage, not an unbiased evaluation or optimal policy. Zero commands can be safe brakes; always-zero label MAE is not in-game quality.'))
    print([dict(source=c['source'], worlds=c['worlds'], totals=c['totals']) for c in corpora], flush=True)


if __name__ == '__main__':
    main()
