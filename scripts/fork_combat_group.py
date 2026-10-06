"""Offline typed group supervision. No live command correction or hidden state."""
import argparse
import json
import math
import pathlib
import time
from fork_combat_maneuver import verified_rows
from fork_combat_exploration import fork
from fork_combat_aim import finish_actor
from ppo_combat import network, torch, sha, training_devices


def labels(f, raw, range_band=False, finish_only=False, barrel_escape=False, obstacle_escape=False):
    if len(f) != 810 or len(raw) != 8 or not all(math.isfinite(v) for v in f + raw):
        raise ValueError('Finite v4 features and eight actor outputs required')
    if sum((range_band, finish_only, barrel_escape, obstacle_escape)) > 1:
        raise ValueError('Separate curriculum modes required')
    if obstacle_escape:
        if f[2] != 1 or f[21] != 1 or not any(f[73+12*i] == 1 for i in range(8)):
            return None, None, None
        # Reconstruct the parent's proposed horizontal direction in current
        # view coordinates, including its simultaneous yaw/pitch command.
        yaw = math.radians(180*math.tanh(raw[2]))
        pitch = max(-89., min(89., math.degrees(math.atan2(f[8], f[9]))+180*math.tanh(raw[3])))
        forward, side = math.tanh(raw[0])*math.cos(math.radians(pitch/3)), math.tanh(raw[1])
        x, y = forward*math.cos(yaw)+side*math.sin(yaw), forward*math.sin(yaw)-side*math.cos(yaw)
        if math.hypot(x,y) < .1:
            return None, None, None
        direction = int(math.floor(math.atan2(y,x)/(math.pi/4)+.5)) % 8
        at = 25+6*direction
        # Unknown probes do not prove a wall. This nearest 45-degree probe is
        # a local hint, not a trace of the continuous proposed action.
        if f[at] != 1 or f[at+2] != 1 or f[at+3]*64 >= 40:
            return None, None, None
    if barrel_escape:
        nearby = f[259] == 1 and any(
            f[260+9*i] == 1 and f[265+9*i] == 1 and
            math.hypot(f[262+9*i],f[263+9*i])*512 < 256
            for i in range(4))
        if not nearby or not any(f[73+12*i] == 1 for i in range(8)):
            return None, None, None
    if finish_only:
        if range_band:
            raise ValueError('Finish-only and general range-band are separate experiments')
        slots = [i for i in range(8) if f[73+12*i] == 1]
        if len(slots) != 1 or f[808]*8 != 1:
            return None, None, None
        i = slots[0]
        if f[82+12*i] != 1 or math.hypot(f[75+12*i], f[76+12*i])*512 <= 320:
            return None, None, None
        # Sole visible is not proof that an occluded enemy is dead.
        return labels(f, raw, range_band=True)
    continuation = range_band and any(
        f[73+12*i] == 1 and (f[83+12*i] == 1 or
        (f[82+12*i] == 1 and math.hypot(f[75+12*i], f[76+12*i])*512 > 256))
        for i in range(8))
    if f[808]*8 < 2 and not continuation and not barrel_escape and not obstacle_escape:
        return None, None, None
    enemies = []
    aim_candidates = []
    for slot in range(8):
        at = 73 + 12*slot
        if f[at] != 1:
            continue
        x, y = f[at+2]*512, f[at+3]*512
        vx, vy = (f[at+6]*400, f[at+7]*400) if f[at+5] == 1 else (0., 0.)
        parasite, gunner = f[at+9] == 1, f[at+10] == 1
        enemies.append((x, y, vx, vy, parasite, gunner))
        bbox = 426 + 5*slot
        if f[at+11] != 1 or f[bbox] != 1:
            continue
        angles = []
        for sine, cosine in ((f[bbox+1], f[bbox+2]), (f[bbox+3], f[bbox+4])):
            if abs(math.hypot(sine, cosine)-1) > 1e-5:
                raise ValueError('Invalid observed bbox angular features')
            angles.append(max(-20., min(20., math.degrees(math.atan2(sine, cosine)))))
        distance = math.hypot(x, y)
        priority = (3*max(0., 1-distance/512) if parasite else
                    1.5*max(0., 1-distance/1024) if gunner else 1.)
        aim_candidates.append((priority, -slot, angles))
    selected = max(aim_candidates) if aim_candidates and not (barrel_escape or obstacle_escape) else None
    aim, slot = (selected[2], -selected[1]) if selected else (None, None)
    if f[2] != 1 or f[21] != 1:
        return None, aim, slot
    barrels = []
    if f[259] == 1:
        for i in range(4):
            at = 260 + 9*i
            if f[at] == 1 and f[at+5] == 1:
                barrels.append((f[at+2]*512, f[at+3]*512))
    projectiles = []
    if f[169] == 1:
        for i in range(4):
            at = 170 + 12*i
            # Unknown velocity does not invent a trajectory. This does not prove safety.
            if f[at] == 1 and f[at+5] == 1 and abs(f[at+4]*512) < 128:
                projectiles.append((f[at+2]*512, f[at+3]*512,
                                    f[at+6]*400, f[at+7]*400))
    candidates = []
    for direction in range(8):
        at = 25 + 6*direction
        if (f[at] != 1 or f[at+2] != 1 or f[at+3]*64 < 40 or
                f[at+4] != 1 or not 0 <= f[at+5]*32 <= 48):
            continue
        x, y = math.cos(direction*math.pi/4), math.sin(direction*math.pi/4)
        if any(abs(bx-40*x) < 48 and abs(by-40*y) < 48 for bx, by in barrels):
            continue
        score = .15*f[at+3] + .05*f[at+1]
        if barrel_escape:
            # Conservative horizontal blast margin, not a prediction of detonation.
            score -= sum(8*max(0.,1-math.hypot(bx-40*x,by-40*y)/256)**2
                         for bx,by in barrels)
        for ex, ey, vx, vy, parasite, gunner in enemies:
            # A local 40-unit step over .25 game seconds; not an engine rollout.
            future = math.hypot(ex+.25*vx-40*x, ey+.25*vy-40*y)
            if parasite:
                score -= 4*max(0., 1-future/320)
                if range_band:
                    score -= 2*max(0., (future-320)/320)
            elif gunner:
                score -= max(0., 1-future/256)
                if range_band:
                    score -= max(0., (future-512)/512)
                distance = math.hypot(ex, ey)
                if distance > 0:
                    score += .6*abs(x*ey-y*ex)/distance
        for px, py, vx, vy in projectiles:
            if px*vx+py*vy >= 0 and math.hypot(px, py) >= 80:
                continue  # Receding shots do not become incoming threats.
            rvx, rvy = vx-160*x, vy-160*y
            speed2 = rvx*rvx+rvy*rvy
            t = max(0., min(.25, -(px*rvx+py*rvy)/speed2)) if speed2 else 0.
            closest = math.hypot(px+t*rvx, py+t*rvy)
            score -= 4*max(0., 1-closest/128)**2
        candidates.append((score, direction))
    if not candidates:
        return None, aim, slot
    direction = max(candidates)[1]
    yaw_step = aim[0] if aim else 180*math.tanh(raw[2])
    pitch_step = aim[1] if aim else 180*math.tanh(raw[3])
    angle = direction*math.pi/4-math.radians(yaw_step)
    pitch = max(-89., min(89., math.degrees(math.atan2(f[8], f[9]))+pitch_step))
    return [max(-.95, min(.95, .75*math.cos(angle)/math.cos(math.radians(pitch/3)))),
            -.75*math.sin(angle)], aim, slot


def fit(model, checkpoint, rows, retain, epochs=2000, range_band=False, finish_only=False, barrel_escape=False, obstacle_escape=False):
    if model.get('feature_version') != 'combat_features_v4' or not 1 <= epochs <= 10000 or len(retain) < 2:
        raise ValueError('Typed model, bounded budget and solo retention required')
    child, state = fork(model, checkpoint, model['log_std'])
    x = torch.tensor([r['features'] for r in rows+retain], dtype=torch.float32)
    if x.ndim != 2 or x.shape[1] != 810 or not torch.isfinite(x).all():
        raise ValueError('Finite typed observations required')
    with torch.no_grad(): old = network(model['actor'])(x)
    target = old[:, :4].clone()
    movement, aiming, choices = [], [], []
    for i, row in enumerate(rows):
        move, aim, slot = labels(row['features'], old[i].tolist(), range_band, finish_only, barrel_escape, obstacle_escape)
        if move is not None:
            target[i, :2] = torch.atanh(torch.tensor(move)); movement.append(i)
        if aim is not None:
            target[i, 2:4] = torch.atanh(torch.tensor(aim)/180); aiming.append(i); choices.append(slot)
    if len(movement) < 2 or (not (barrel_escape or obstacle_escape) and len(aiming) < 2):
        raise ValueError('At least two observed group movement/aim labels required')
    weights = torch.ones_like(target); weights[:, 2:4] = 40
    weights[len(rows):, :2] = 10
    active = set(movement+aiming)
    inactive = [i for i in range(len(rows)) if i not in active]
    if finish_only or barrel_escape or obstacle_escape:
        weights[inactive, :4] *= 5
        weights[movement, :2] *= 100
        weights[aiming, 2:4] *= 100
    if barrel_escape or obstacle_escape:
        # Preserve finishing behavior outside the newly supervised blast states.
        weights[inactive, :4] *= 10
        weights[movement, :2] /= 5
    if obstacle_escape:
        # Local escape labels must not rewrite solo combat or shared aim heads.
        weights[inactive, :2] *= 2
        weights[len(rows):, :2] *= 10
        weights[:, 2:4] *= 10
    benchmark = {}
    def setup(device):
        actor = network(child['actor']).to(device)
        optimizer = torch.optim.Adam(actor.parameters(), lr=.0003)
        inputs, desired, teacher, importance = x.to(device), target.to(device), old.to(device), weights.to(device)
        def step():
            optimizer.zero_grad(); raw = actor(inputs)
            loss = ((raw[:, :4]-desired).square()*importance).mean() + (100 if obstacle_escape else 10)*(raw[:, 4:]-teacher[:, 4:]).square().mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite group fit')
            loss.backward(); optimizer.step()
        return actor, step
    for device in training_devices():
        actor, step = setup(device)
        for _ in range(10): step()
        if device == 'cuda': torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(30): step()
        if device == 'cuda': torch.cuda.synchronize()
        benchmark[device] = (time.perf_counter()-start)/30
    device = min(benchmark, key=benchmark.get); actor, step = setup(device)
    for _ in range(epochs): step()
    actor = actor.cpu()
    with torch.no_grad():
        new = actor(x); solo = (new[len(rows):, :4].tanh()-old[len(rows):, :4].tanh()).abs().mean(0)
        attack = (new[:, 4].sigmoid()-old[:, 4].sigmoid()).abs().mean()
        a, b = old[:, 5:].log_softmax(1), new[:, 5:].log_softmax(1)
        vertical = (a.exp()*(a-b)).sum(1).mean()
        if solo[:2].max() > .05 or solo[2:].max()*180 > 2 or attack > .05 or vertical > .02:
            raise ValueError(f'Group training-state retention gate failed: solo={solo.tolist()}, attack={float(attack)}, vertical={float(vertical)}')
        inactive_drift = (new[inactive,:4].tanh()-old[inactive,:4].tanh()).abs().mean(0) if inactive else torch.zeros(4)
        if (finish_only or barrel_escape or obstacle_escape) and (inactive_drift[:2].max() > .05 or inactive_drift[2:].max()*180 > 2):
            raise ValueError(f'Inactive-state retention gate failed: {inactive_drift.tolist()}')
        report = dict(rows=len(rows),retention_rows=len(retain),epochs=epochs,range_band=range_band,finish_only=finish_only,barrel_escape=barrel_escape,obstacle_escape=obstacle_escape,inactive_rows=len(inactive),inactive_movement_mae=inactive_drift[:2].tolist(),inactive_aim_mae_degrees=(inactive_drift[2:]*180).tolist(),movement_labels=len(movement),aim_labels=len(aiming),nonnearest_targets=sum(s!=0 for s in choices),device=device,benchmark_seconds_per_step=benchmark,
                      movement_mae_before=(old[movement,:2].tanh()-target[movement,:2].tanh()).abs().mean(0).tolist(),movement_mae_after=(new[movement,:2].tanh()-target[movement,:2].tanh()).abs().mean(0).tolist(),
                      aim_mae_before_degrees=((old[aiming,2:4].tanh()-target[aiming,2:4].tanh()).abs().mean(0)*180).tolist() if aiming else None,aim_mae_after_degrees=((new[aiming,2:4].tanh()-target[aiming,2:4].tanh()).abs().mean(0)*180).tolist() if aiming else None,solo_movement_mae=solo[:2].tolist(),solo_aim_mae_degrees=(solo[2:]*180).tolist(),attack_probability_mae=float(attack),vertical_kl=float(vertical))
    finish_actor(child, state, actor)
    return child, state, report


def main():
    ap = argparse.ArgumentParser()
    for name in ('model','checkpoint','out'): ap.add_argument('--'+name,type=pathlib.Path,required=True)
    for name in ('rollout','retain-rollout'): ap.add_argument('--'+name,type=pathlib.Path,action='append',required=True)
    ap.add_argument('--epochs',type=int,default=2000)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--range-band',action='store_true')
    mode.add_argument('--finish-only',action='store_true')
    mode.add_argument('--barrel-escape',action='store_true')
    mode.add_argument('--obstacle-escape',action='store_true'); a = ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(20261006);torch.use_deterministic_algorithms(True)
    receipts = {str(p):sha(p) for p in (a.model,a.checkpoint)}
    state = torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if state.get('weights_sha256') != sha(a.model): raise ValueError('Parent checkpoint differs')
    rows, retain = verified_rows(a.rollout,receipts), verified_rows(a.retain_rollout,receipts)
    if {r['seed'] for r in rows} & {r['seed'] for r in retain}: raise ValueError('Retention overlaps group training')
    child,state,report = fit(json.loads(a.model.read_text()),state,rows,retain,a.epochs,a.range_band,a.finish_only,a.barrel_escape,a.obstacle_escape)
    if any(sha(pathlib.Path(p)) != digest for p,digest in receipts.items()): raise ValueError('Training source changed')
    a.out.mkdir(exist_ok=False);weights = a.out/'weights.json';weights.write_text(json.dumps(child,allow_nan=False));state['weights_sha256']=sha(weights);torch.save(state,a.out/'checkpoint.pt')
    report.update(version='combat_group_fork_v1',source_sha256=receipts,weights_sha256=sha(weights),script_sha256=sha(pathlib.Path(__file__)),training_seeds=sorted({r['seed'] for r in rows}),retention_seeds=sorted({r['seed'] for r in retain}),scope='Offline local group movement and bbox aim labels using current v4 features only; no hidden state or live correction. Not PPO. .25s linear hint is not engine dynamics or a guaranteed safe path. Solo/attack/vertical retained on training states only; both Adam states reset, critic output zero, std/RNG/history/counters retained. Fresh rollouts required for subsequent PPO.')
    (a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))


if __name__ == '__main__': main()
