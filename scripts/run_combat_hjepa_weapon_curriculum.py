"""Collect three fresh on-policy weapon batches, carrying the frozen actor's Adam state."""
import argparse
import copy
import json
import os
import pathlib
import shutil
import subprocess
import sys
from train_combat_hjepa import save, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    ap.add_argument('--previous-batch',type=pathlib.Path,required=True)
    ap.add_argument('--template-plan',type=pathlib.Path,required=True)
    ap.add_argument('--seed-offset',type=int,default=433)
    ap.add_argument('--port',type=int,default=39300)
    a = ap.parse_args()
    if not os.environ.get('Q2COOPBOT_DIAGNOSTIC_ENGINE_ROOT'):
        raise ValueError('explicit inventory-capable native engine required')
    host = shutil.which('pwsh')
    if not host:
        raise ValueError('PowerShell 7 required')
    previous = a.previous_batch.resolve()
    template = json.loads(a.template_plan.read_text())
    ids = ','.join(dict.fromkeys(task['episode']['id'] for task in template['tasks']))
    a.out = a.out.resolve()
    a.out.mkdir(parents=True,exist_ok=False)
    scripts = ['train_combat_hjepa_weapon_onpolicy.py','train_combat_hjepa_branch_preferences.py',
               'train_combat_hjepa.py','run_combat_hjepa_weapon_curriculum.py']
    sources = {str((pathlib.Path('scripts')/name).resolve()):sha(pathlib.Path('scripts')/name) for name in scripts}
    source_dir = a.out/'source';source_dir.mkdir()
    for name in scripts:
        shutil.copyfile(pathlib.Path('scripts')/name,source_dir/name)
    save(a.out/'protocol.json',dict(rounds=3,worlds_per_round=16,streams=4,updates_per_round=4,
            seed_offsets=list(range(a.seed_offset,a.seed_offset+3)),previous_checkpoint=str(previous),
            previous_checkpoint_sha256=sha(previous),source_sha256=sources,
            fixed_evaluation='test490/491 after all three batches; never fit on viewed TEST',promotion=False))
    def run(args):
        for name,digest in sources.items():
            if sha(pathlib.Path(name)) != digest:
                raise ValueError('curriculum source changed')
        subprocess.run([str(v) for v in args],check=True)
    rounds = []
    for index in range(3):
        root = a.out/f'round-{index+1}';root.mkdir()
        parent_path = previous.parent/'model.json'
        parent = json.loads(parent_path.read_text())
        if parent.get('weapon_sampling_seed') is not None:
            raise ValueError('previous result must be greedy evaluation weights')
        variants=[]
        for stream in range(4):
            model = copy.deepcopy(parent)
            model['weapon_sampling_seed'] = 261300+10*index+stream
            path = root/f'behavior-{stream}.json'
            save(path,model)
            variants.append(dict(name=f'stream-{stream}',mode='learned',model=str(path),
                                 sha256=sha(path),contract='stock_weapons_fire_v1'))
        save(root/'models.json',variants)
        run(['go','run','./cmd/q2episode','--episodes',ids,'--split','train','--mode','rules',
             '--count','1','--seed-offset',a.seed_offset+index,'--out',root/'plan.json','--artifacts',root/'runtime'])
        print(f'ROUND {index+1}: collecting 64 fresh native episodes',flush=True)
        run([host,'-NoProfile','-File','scripts/run_combat_hjepa_matrix.ps1',
             '-Plan',root/'plan.json','-Models',root/'models.json','-OutputRoot',root/'native','-Port',a.port])
        data=[]
        for stream in range(4):
            path=root/f'rollout-{stream}.jsonl';data += ['--data',path]
            run(['go','run','./cmd/q2hjepa-weapon-rollout','--root',root/'native',
                 '--model',root/f'behavior-{stream}.json','--variant',f'stream-{stream}','--out',path])
        run([sys.executable,'scripts/train_combat_hjepa_weapon_onpolicy.py',*data,
             '--previous-batch',previous,'--steps','4','--out',root/'fit'])
        run(['go','run','./cmd/q2hjepa-check','--model',root/'fit/model.json',
             '--parity',root/'fit/parity-vectors.json','--weapon-only'])
        report=json.loads((root/'fit/report.json').read_text())
        rounds.append(dict(round=index+1,rows=report['rows'],episodes=report['episodes'],
                           total_steps=report['total_steps'],model_sha256=report['model_sha256']))
        save(root/'complete.json',rounds[-1])
        previous=root/'fit/checkpoint.pt'
        print(f'ROUND {index+1} COMPLETE: {json.dumps(rounds[-1])}',flush=True)
    save(a.out/'complete.json',dict(state='complete',rounds=rounds,checkpoint=str(previous),
             model=str(previous.parent/'model.json'),model_sha256=sha(previous.parent/'model.json'),promotion=False))


if __name__=='__main__':
    main()
