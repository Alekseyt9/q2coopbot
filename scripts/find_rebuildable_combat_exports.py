"""List generated PPO export streams; preserve native captures and model files."""
import argparse,json,os,pathlib,re

ap=argparse.ArgumentParser()
ap.add_argument('--root',type=pathlib.Path,required=True)
ap.add_argument('--out',type=pathlib.Path,required=True)
args=ap.parse_args();root=args.root.resolve();removed=[];exports=[]
for base,dirs,names in os.walk(root,followlinks=False):
    dirs[:]=[d for d in dirs if not (pathlib.Path(base)/d).is_junction() and not (pathlib.Path(base)/d).is_symlink()]
    if 'rollout.jsonl' not in names or 'report.json' not in names:continue
    folder=pathlib.Path(base)
    try:report=json.loads((folder/'report.json').read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):continue
    if report.get('version')!='combat_ppo_rollout_v1':continue
    exports.append(str(folder))
    candidates=[folder/n for n in ['rollout.jsonl','sequence.jsonl'] if (folder/n).is_file()]
    for child in folder.iterdir():
        if child.is_dir() and re.fullmatch(r'replay-\d+',child.name):
            candidates.extend(child/n for n in ['steps.jsonl','rewards.jsonl','server_outcomes.jsonl'] if (child/n).is_file())
    for p in candidates:
        assert p.resolve().is_relative_to(root) and not p.is_symlink()
        s=p.stat();removed.append(dict(path=str(p),bytes=s.st_size,links=s.st_nlink,export=str(folder)))
plan=dict(version='rebuildable_combat_export_cleanup_v1',root=str(root),files=removed,exports=exports,
          bytes=sum(x['bytes'] for x in removed),scope='Generated PPO arrays and replay projections only; native captures, models, checkpoints and reports retained.')
args.out.write_text(json.dumps(plan,indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(removed),bytes=plan['bytes'],exports=len(exports))))
