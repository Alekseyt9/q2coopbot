"""Change only the pinned curriculum contract of a decision-scope checkpoint."""
import argparse
import copy
from pathlib import Path
import shutil

from evaluate_registered_combat_models import read, save, sha
from train_combat_bc import torch, training_devices
from ppo_recurrent import durable_write
from combat_checkpoint_scope import DECISION_SCOPE, require_checkpoint_scope
from fork_conditional_weapon_exploration_cuda import equal_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','old-protocol','protocol','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    pins = {}

    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path)
        assert expected is None or digest==expected,str(path)
        pins[str(path)]=digest
        return path

    pin(__file__)
    seal=read(pin(args.source/'complete.json'))
    for name,key in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):
        pin(args.source/name,seal[key])
    old,new=read(pin(args.old_protocol)),read(pin(args.protocol))
    assert old['version']==new['version']==DECISION_SCOPE
    allowed={'episodes','quality_endpoint','curriculum_change'}
    assert {key:value for key,value in old.items() if key not in allowed}=={key:value for key,value in new.items() if key not in allowed}
    assert len(new['episodes'])==8 and all('-gunner-' in name for name in new['episodes'])
    training_devices()
    cp=torch.load(args.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    require_checkpoint_scope(cp,DECISION_SCOPE)
    assert cp['weights_sha256']==seal['weights_sha256'] and cp['decision_scope']['protocol_sha256']==sha(args.old_protocol)
    assert cp['decision_scope']['updates_completed']==2
    # Host load/save is byte transport only; tensor checks below run on CUDA.
    child=torch.load(args.source/'checkpoint.pt',map_location='cpu',weights_only=True)
    child['decision_scope']['protocol_sha256']=sha(args.protocol)
    record=dict(version='combat_decision_curriculum_contract_fork_v1',source_checkpoint_sha256=seal['checkpoint_sha256'],
                from_protocol_sha256=sha(args.old_protocol),to_protocol_sha256=sha(args.protocol),training_performed=False)
    child['decision_curriculum_forks']=child.get('decision_curriculum_forks',[])+[record]
    args.out.mkdir(parents=True)
    shutil.copy2(args.source/'weights.json',args.out/'weights.json')
    durable_write(args.out/'checkpoint.pt',lambda stream:torch.save(child,stream))
    loaded=torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    assert set(loaded)==set(cp)|{'decision_curriculum_forks'}
    for key in cp:
        if key not in ('decision_scope','decision_curriculum_forks'):
            assert equal_tree(cp[key],loaded[key]),key
    restored=copy.deepcopy(loaded['decision_scope']);restored['protocol_sha256']=cp['decision_scope']['protocol_sha256']
    assert restored==cp['decision_scope']
    assert loaded['decision_curriculum_forks'][:-1]==cp.get('decision_curriculum_forks',[])
    assert sha(args.out/'weights.json')==seal['weights_sha256']
    assert all(sha(path)==digest for path,digest in pins.items())
    save(args.out/'report.json',dict(state='complete',version=record['version'],device='cuda',inputs=pins,
        source_weights_sha256=seal['weights_sha256'],weights_bytes_exact=True,all_neural_optimizer_rng_history_exact=True,
        checkpoint_roundtrip_exact=True,training_performed=False,promotion=False,protocol_sha256=sha(args.protocol)))
    save(args.out/'complete.json',dict(state='complete',weights_sha256=sha(args.out/'weights.json'),
        checkpoint_sha256=sha(args.out/'checkpoint.pt'),report_sha256=sha(args.out/'report.json')))
    print(dict(state='complete',weights_sha256=seal['weights_sha256'],all_neural_optimizer_rng_history_exact=True))


if __name__=='__main__':
    main()
