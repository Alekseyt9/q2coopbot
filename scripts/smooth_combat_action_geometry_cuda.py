"""Recompute version2 soft geometry from compact causal inputs; no raw replay."""
import argparse
import shutil
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();assert not args.out.exists()
    seal=read(args.source/'complete.json');report=read(args.source/'report.json')
    assert seal['state']==report['state']=='complete' and sha(args.source/'report.json')==seal['report_sha256']
    assert sha(args.source/'dataset.pt')==report['dataset_sha256'] and sha(args.source/'members.jsonl.gz')==report['members_sha256']
    from ppo_combat import torch,training_devices
    training_devices();torch.set_default_device('cuda')
    from combat_action_geometry import features_for_version,names_for_version,SMOOTH_VERSION
    tensors=torch.load(args.source/'dataset.pt',map_location='cuda',weights_only=True)
    old=tensors['action_geometry'].clone();extra=features_for_version(tensors['features_with_recoil'],SMOOTH_VERSION)
    continuous=[i for i in range(29) if i not in (15,28)]
    assert torch.equal(old[:,continuous],extra[:,continuous])
    tensors.update(action_geometry=extra,geometry_version=SMOOTH_VERSION,geometry_names=list(names_for_version(SMOOTH_VERSION)))
    args.out.mkdir(parents=True);torch.save(tensors,args.out/'dataset.pt')
    shutil.copyfile(args.source/'members.jsonl.gz',args.out/'members.jsonl.gz')
    report.update(version='combat_shot_prediction_dataset_v4',geometry_version=SMOOTH_VERSION,
        dataset_sha256=sha(args.out/'dataset.pt'),members_sha256=sha(args.out/'members.jsonl.gz'),
        source_complete_sha256=sha(args.source/'complete.json'),geometry_helper_sha256=sha(Path(__file__).with_name('combat_action_geometry.py')),
        script_sha256=sha(__file__),scope='Same compact TRAIN inputs, labels, members, cases and folds; only bbox intersection features15/28 replaced by clamp((ray slab gap+1)/2,0,1), a +/-1 world-unit band. No raw traces copied or replayed.')
    save(args.out/'report.json',report)
    save(args.out/'complete.json',dict(state='complete',report_sha256=sha(args.out/'report.json'),dataset_sha256=report['dataset_sha256'],policy_training=False,ppo_ready=False))
    print('Soft geometry v2:',len(extra),'shots; labels/folds/base features retained')


if __name__=='__main__':main()
