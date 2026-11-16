"""Three source-bound on-policy primitive batches with categorical exploration.

The protocol and disjoint verified plans must exist before execution. Resume
only completed stages; an incomplete native/fit directory requires inspection,
never an automatic restart of a possibly live process.
"""
import argparse
import concurrent.futures
import copy
import json
import os
import pathlib
import shutil
import subprocess
import sys
import torch
from train_combat_hjepa import save, sha


def same(x, y):
    if isinstance(x, torch.Tensor):
        return isinstance(y, torch.Tensor) and torch.equal(x, y)
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    if isinstance(x, (tuple, list)):
        return len(x) == len(y) and all(same(a, b) for a, b in zip(x, y))
    return x == y


def immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('existing stage input changed: '+str(path))
    else:
        save(path, value)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=pathlib.Path, required=True)
    ap.add_argument('--port', type=int, default=39100)
    a = ap.parse_args(); a.root = a.root.resolve()
    if not os.environ.get('Q2COOPBOT_DIAGNOSTIC_ENGINE_ROOT') or not shutil.which('pwsh'):
        raise ValueError('explicit native engine and PowerShell7 required')
    protocol_path = a.root/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if (protocol['version'] != 'hjepa_categorical_curriculum_v1' or protocol['rounds'] != 3
            or protocol['streams'] != 4 or protocol['temperature'] != 1.):
        raise ValueError('frozen categorical curriculum protocol required')
    parent_path = pathlib.Path(protocol['parent'])
    if sha(parent_path) != protocol['parent_sha256']:
        raise ValueError('initial source changed')
    parent = json.loads(parent_path.read_text())
    if parent.get('motor') or parent['planner'] != 'bc':
        raise ValueError('shared primitive actor required')
    plan_seeds = set()
    for plan in protocol['plans'] + [protocol['validation']]:
        path = pathlib.Path(plan['path'])
        if sha(path) != plan['sha256'] or plan_seeds & set(plan['seeds']):
            raise ValueError('plan changed or worlds reused')
        plan_seeds.update(plan['seeds'])
    source_names = ['run_combat_hjepa_actor_curriculum.py','train_combat_hjepa_actor_natural.py',
                    'train_combat_hjepa_actor_ppo.py','hjepa_weapon_returns.py','train_combat_hjepa.py',
                    'audit_combat_hjepa_actor_rollout.py','audit_combat_hjepa_weapons.py',
                    'report_combat_hjepa_categorical.py','report_combat_hjepa_weapon_stochastic.py',
                    'train_combat_hjepa_weapon_onpolicy.py']
    sources = {name:sha(pathlib.Path('scripts')/name) for name in source_names}
    immutable_json(a.root/'source-manifest.json', sources)
    snapshot = a.root/'source'; snapshot.mkdir(exist_ok=True)
    for name in source_names:
        path = snapshot/name
        if path.exists() and sha(path) != sources[name]:
            raise ValueError('snapshot changed')
        if not path.exists():
            shutil.copyfile(pathlib.Path('scripts')/name, path)
    protocol_sha = sha(protocol_path)

    def run(args):
        if sha(protocol_path) != protocol_sha or any(sha(pathlib.Path('scripts')/n) != d for n, d in sources.items()):
            raise ValueError('source/protocol changed during curriculum')
        subprocess.run([str(v) for v in args], check=True)

    def matrix(plan, models, out):
        if not (out/'complete.json').exists():
            if out.exists():
                raise ValueError('incomplete matrix requires authoritative process inspection: '+str(out))
            run(['pwsh','-NoProfile','-File','scripts/run_combat_hjepa_matrix.ps1',
                 '-Plan',plan,'-Models',models,'-OutputRoot',out,'-Port',a.port])
        complete = json.loads((out/'complete.json').read_text())
        if complete['state'] != 'complete' or complete['source_plan_sha256'] != sha(plan):
            raise ValueError('matrix completion/plan mismatch')
        expected = json.loads(pathlib.Path(models).read_text())
        if [(v['name'],v.get('sha256')) for v in complete['models']] != [(v['name'],v.get('sha256')) for v in expected]:
            raise ValueError('matrix model binding mismatch')

    hot = copy.deepcopy(parent); hot['primitive_sampling']['categorical_temperature'] = 1.
    hot['primitive_sampling']['seed'] = 262000
    immutable_json(a.root/'initial-hot.json', hot)
    rounds, used_seeds = [], set()
    current = a.root/'initial-hot.json'
    for index, plan in enumerate(protocol['plans']):
        root = a.root/f'round-{index+1}'; root.mkdir(exist_ok=True)
        source = json.loads(current.read_text())
        if source['primitive_sampling']['categorical_temperature'] != 1. or set(plan['seeds']) & used_seeds:
            raise ValueError('changed temperature or reused TRAIN worlds')
        used_seeds.update(plan['seeds'])
        variants = []
        for stream in range(4):
            model = copy.deepcopy(source); model['primitive_sampling']['seed'] = 262000+index*4+stream
            path = root/f'behavior-{stream}.json'; immutable_json(path, model)
            variants.append(dict(name=f'stream-{stream}',mode='learned',model=str(path),sha256=sha(path),contract='stock_weapons_fire_v1'))
        immutable_json(root/'models.json', variants)
        immutable_json(root/'parent.json',dict(path=str(current),sha256=sha(current),used_seeds_before=sorted(used_seeds-set(plan['seeds']))))
        save(a.root/'state.json',dict(stage='collect_train',round=index+1,completed_rounds=len(rounds)))
        print(f'ROUND{index+1}: collect64 native TRAIN games, offset{plan["offset"]}',flush=True)
        matrix(pathlib.Path(plan['path']),root/'models.json',root/'native')

        def export_audit(stream):
            data = root/f'rollout-{stream}.jsonl'
            if not data.exists():
                run(['go','run','./cmd/q2hjepa-actor-rollout','-root',root/'native',
                     '-model',root/f'behavior-{stream}.json','-variant',f'stream-{stream}','-out',data])
            audit_path = root/f'likelihood-audit-{stream}.json'
            if not audit_path.exists():
                run([sys.executable,'scripts/audit_combat_hjepa_actor_rollout.py','--data',data,'--out',audit_path])
            audit = json.loads(audit_path.read_text())
            if not audit['passed'] or audit['data_sha256'] != sha(data):
                raise ValueError('likelihood audit failed/stale')
            return data

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            data = list(pool.map(export_audit,range(4)))
        data_args = [arg for path in data for arg in ['--data',path]]
        fit = root/'fit'
        save(a.root/'state.json',dict(stage='fit',round=index+1,completed_rounds=len(rounds)))
        if not (fit/'report.json').exists():
            if fit.exists():
                raise ValueError('incomplete fit requires process inspection')
            run([sys.executable,'scripts/train_combat_hjepa_actor_natural.py',*data_args,'--out',fit])
        report = json.loads((fit/'report.json').read_text())
        if report['stopped'] or len(report['history']) != 4 or report['model_sha256'] != sha(fit/'model.json'):
            raise ValueError('four-update fit did not pass')
        fitted = json.loads((fit/'model.json').read_text())
        frozen = copy.deepcopy(fitted)
        frozen['levels'][0]['encoder'] = source['levels'][0]['encoder']
        frozen['actor'] = source['actor']
        frozen['primitive_sampling']['log_std'] = source['primitive_sampling']['log_std']
        frozen['primitive_sampling']['seed'] = source['primitive_sampling']['seed']
        if frozen != source:
            raise ValueError('nonprimitive component changed')
        run(['go','run','./cmd/q2hjepa-check','--model',fit/'model.json','--parity',fit/'parity.json','--motor-only'])
        if index == 0 and not (root/'resume-check.json').exists():
            run([sys.executable,'scripts/train_combat_hjepa_actor_natural.py',*data_args,'--out',root/'split-2','--updates','2'])
            run([sys.executable,'scripts/train_combat_hjepa_actor_natural.py',*data_args,'--out',root/'resumed-4','--resume',root/'split-2/checkpoint.pt'])
            full = torch.load(fit/'checkpoint.pt',map_location='cpu',weights_only=False)
            resumed = torch.load(root/'resumed-4/checkpoint.pt',map_location='cpu',weights_only=False)
            checks = {k:same(full[k],resumed[k]) for k in ['state','history','binding','parent_fingerprint','algorithm']}
            checks['model_bytes'] = sha(fit/'model.json') == sha(root/'resumed-4/model.json')
            if not all(checks.values()):
                raise ValueError('temperature1 exact resume failed')
            save(root/'resume-check.json',dict(passed=True,checks=checks))
        current = fit/'model.json'
        round_info = dict(round=index+1,offset=plan['offset'],rows=report['admission']['rows'],admitted=report['admission']['admitted'],
                          parent_sha256=sha(pathlib.Path(json.loads((root/'parent.json').read_text())['path'])),model_sha256=sha(current),
                          checkpoint_sha256=sha(fit/'checkpoint.pt'),mean_kl=report['history'][-1]['mean_kl'])
        immutable_json(root/'complete.json',round_info); rounds.append(round_info)
        save(a.root/'training-progress.json',dict(rounds=rounds,used_train_seeds=sorted(used_seeds)))
        print('ROUND COMPLETE '+json.dumps(round_info),flush=True)

    final = json.loads(current.read_text()); variants = []
    for role, model in [('cold',parent),('hot',hot),('trained',final)]:
        for stream, seed in enumerate([262500,262501]):
            spec = copy.deepcopy(model);spec['primitive_sampling']['seed'] = seed
            path = a.root/f'eval-{role}-s{stream}.json';immutable_json(path,spec)
            variants.append(dict(name=f'{role}-s{stream}',mode='learned',model=str(path),sha256=sha(path),contract='stock_weapons_fire_v1'))
    for role, model in [('source-mean',parent),('trained-mean',final)]:
        spec = copy.deepcopy(model);spec.pop('primitive_sampling')
        path = a.root/f'eval-{role}.json';immutable_json(path,spec)
        variants.append(dict(name=role,mode='learned',model=str(path),sha256=sha(path),contract='stock_weapons_fire_v1'))
    u1 = pathlib.Path('workspace/runtime/live-coop-20261027-192152/u1-weights.json').resolve()
    if sha(u1) != '21c3b680e0b6090d601d524f6da53d73ae6dd083c8e7a9a6da2e6d268ccadd26':
        raise ValueError('U1 baseline changed')
    variants.append(dict(name='live-u1',mode='learned',model=str(u1),sha256=sha(u1),contract='stock_weapons_fire_v1'))
    immutable_json(a.root/'models-validation.json',variants)
    immutable_json(a.root/'models-rules.json',[dict(name='rules',mode='rules')])
    save(a.root/'state.json',dict(stage='validation',round=3,completed_rounds=3))
    matrix(pathlib.Path(protocol['validation']['path']),a.root/'models-validation.json',a.root/'native-validation')
    matrix(pathlib.Path(protocol['validation']['path']),a.root/'models-rules.json',a.root/'rules-validation')
    run([sys.executable,'scripts/report_combat_hjepa_categorical.py','--root',a.root])
    save(a.root/'complete.json',dict(state='complete',rounds=rounds,model=str(current),model_sha256=sha(current),
        native_screen_sha256=sha(a.root/'native-screen.json'),protocol_sha256=protocol_sha,promotion=False))
    save(a.root/'state.json',dict(stage='complete',completed_rounds=3))


if __name__ == '__main__':
    main()
