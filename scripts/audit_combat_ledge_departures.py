"""TRAIN-only factual departure audit; does not create counterfactual labels."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha


def warning(obs, action):
    geometry = obs.get('local_geometry')
    if not geometry or not obs['on_ground'] or geometry.get('frame')!=obs['identity']['frame']:
        return None
    probes = geometry.get('probes') or []
    if len(probes) != 8:
        return None
    yaw = obs['view_angles'][1] * 2 * math.pi / 65536
    vx, vy, _ = obs['velocity']
    if math.hypot(vx, vy) > 25:
        forward = vx*math.cos(yaw) + vy*math.sin(yaw)
        left = -vx*math.sin(yaw) + vy*math.cos(yaw)
        kind = 'observed_velocity'
    else:
        forward, left = action['forward'], -action['side']
        kind = 'applied_move'
    if math.hypot(forward, left) < .01:
        return None
    angle = math.degrees(math.atan2(left, forward)) % 360
    if kind=='applied_move':
        angle=(angle+action.get('yaw_delta_degrees',0))%360
    probe = min(probes, key=lambda p: abs((p['yaw_offset_degrees']-angle+180)%360-180))
    return dict(direction_source=kind, direction_degrees=angle,
                probe_degrees=probe['yaw_offset_degrees'],
                ground_missing=probe.get('ground_drop_at_40') is None,
                standing_clearance=probe.get('standing_hull_clearance'))


def departures(rows, diagnostics, horizon=12):
    events = []
    for i, row in enumerate(rows):
        obs = row['observation']; nxt = row.get('next_observation')
        if not obs['on_ground'] or not nxt or nxt['on_ground']:
            continue
        frame = obs['identity']['frame']; initial_z = obs['position'][2]
        minimum_z = initial_z; count = 0; landed = False
        for j in range(i, min(i+horizon, len(rows))):
            current = rows[j]; before = current['observation']; after = current.get('next_observation')
            if before['identity']['frame'] != frame+count or not after:
                break
            if after['identity']['frame'] != frame+count+1:
                break
            # Every observed successor must still belong to the same life.
            if any(after['identity'][k] != obs['identity'][k] for k in ['map','connection','spawncount','actor','life']):
                break
            count += 1; minimum_z = min(minimum_z, after['position'][2])
            if after['on_ground']:
                landed = True; break
            if current['terminal'] or current['truncated'] or after['health'] <= 0:
                break
        prior = []
        for earlier in rows[max(0,i-5):i+1]:
            age = frame-earlier['observation']['identity']['frame']
            if 0 <= age <= 5:
                signal = warning(earlier['observation'], earlier['applied_action'])
                if signal is not None: prior.append(dict(age_frames=age, **signal))
        diagnostic = diagnostics[frame]; prediction = diagnostic['prediction']
        events.append(dict(frame=frame, index=row['index'], initial_z=initial_z,
             drop=initial_z-minimum_z, deep_drop=initial_z-minimum_z >= 96,
             future_steps=count, landed=landed, right_censored=not landed and count<horizon,
             vertical=row['applied_action']['vertical'], ground_surface=diagnostic['ground_surface'],
             prediction_present=prediction is not None,
             prediction_supported=bool(prediction and prediction['initial_grounded'] and not prediction['support_ambiguous']),
             precommand_warning=warning(obs,row['applied_action']), prior5_warnings=prior))
    return events


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); assert not args.out.exists()
    corpus=args.corpus.resolve(); pool_path=corpus/'pool/report.json'; pool=read(pool_path)
    assert pool['state']=='complete' and pool['source_unchanged']
    pins={str(pool_path):sha(pool_path)}; cases=[]; totals=Counter()
    for job in pool['jobs']:
        if job['plan_index']!=0: continue
        capture_path=Path(job['root'])/'report.json'; capture=read(capture_path)
        worker=Path(capture['results'][0]['root'])
        steps=corpus/'processing-train'/('case-%04d'%job['job'])/'native/replay-0/steps.jsonl'
        trace=worker/'bot.jsonl'
        for path in [capture_path,steps,trace]: pins[str(path)]=sha(path)
        rows=[]
        with steps.open(encoding='utf-8-sig') as stream:
            for line in stream:
                row=json.loads(line); execution=row.get('server_execution') or {}
                if row.get('owner')=='provider' and row['observation']['health']>0 and execution.get('matched') and execution.get('window_exclusive'):
                    rows.append(row)
        selected={r['observation']['identity']['frame']:r for r in rows}; assert len(selected)==len(rows)
        diagnostics={}
        with trace.open(encoding='utf-8-sig') as stream:
            for line in stream:
                raw=json.loads(line); policy=raw.get('combat_policy') or {}; obs=policy.get('observation')
                if not obs or obs['identity']['frame'] not in selected: continue
                row=selected[obs['identity']['frame']]
                assert obs==row['observation'] and raw['sent_command']==row['sent_command']
                assert raw['client_sequence']==row['client_sequence'] and obs['identity']['frame'] not in diagnostics
                diagnostics[obs['identity']['frame']]=dict(prediction=raw.get('pmove_ground_prediction'),ground_surface=raw.get('ground_surface'))
        assert len(diagnostics)==len(rows)
        events=departures(rows,diagnostics)
        for event in events:
            totals['departures']+=1
            if event['deep_drop']:
                totals['deep_drops']+=1; totals['deep_'+event['vertical']]+=1
                totals['deep_supported_prediction']+=event['prediction_supported']
                totals['deep_precommand_warning']+=bool(event['precommand_warning'] and event['precommand_warning']['ground_missing'])
                totals['deep_prior5_warning']+=any(p['ground_missing'] for p in event['prior5_warnings'])
        cases.append(dict(job=job['job'],seed=job['seed'],map=rows[0]['observation']['identity']['map'],rows=len(rows),events=events))
    expected=[j for j in pool['jobs'] if j['plan_index']==0]
    assert cases and len(cases)==len(expected)==len({c['seed'] for c in cases})
    for path,digest in pins.items(): assert sha(Path(path))==digest
    save(args.out,dict(state='complete',version='combat_ledge_departure_audit_v1',split='TRAIN',cases=cases,totals=dict(totals),
         horizon_frames=12,deep_drop_threshold=96,inputs=[dict(path=p,sha256=d) for p,d in pins.items()],
         script_sha256=sha(__file__),training_performed=False,promotion=False,
         scope='Factual executed departure plus following executed commands, not single-action causality. Missing ground at40 within32 means unknown/deep support, not a proven unsafe action. No counterfactual labels or runtime changes.'))
    print(dict(totals))


if __name__=='__main__':main()
