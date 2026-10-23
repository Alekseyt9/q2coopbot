"""Seal inference-only categorical execution forks without changing learned tensors."""
import argparse
import copy
from pathlib import Path
from evaluate_registered_combat_models import read,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--mode',choices=['greedy_target_mode_v1','greedy_target_mode_fire_v1','geometry_shot_rank_aim_v1','geometry_aim_explore_v1'],required=True)
    ap.add_argument('--predictor',type=Path)
    ap.add_argument('--geometry-proof',type=Path)
    args=ap.parse_args();assert not args.out.exists()
    from ppo_combat import torch,training_devices
    training_devices();torch.set_default_device('cuda')
    from ppo_recurrent import durable_json,durable_write
    from fork_conditional_weapon_exploration_cuda import equal_tree
    from migrate_combat_navigation_cuda import branch
    from combat_precision_head import validate_precision_model
    seal=read(args.source/'complete.json');pins={}
    for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
        path=args.source/name;assert sha(path)==seal[key];pins[str(path.resolve())]=seal[key]
    source=read(args.source/'weights.json')
    assert not source.get('execution_mode') and not source['deterministic']
    cp=torch.load(args.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    assert cp['weights_sha256']==seal['weights_sha256']
    for key in ('actor','value'):assert equal_tree(branch(source,key).state_dict(),cp[key])
    model=copy.deepcopy(source);model['execution_mode']=args.mode
    if args.mode=='geometry_shot_rank_aim_v1':
        assert args.predictor and args.geometry_proof
        for folder in (args.predictor,args.geometry_proof):
            closed=read(folder/'complete.json');assert closed['state']=='complete'
            assert sha(folder/'report.json')==closed['report_sha256']
            pins[str((folder/'complete.json').resolve())]=sha(folder/'complete.json')
            pins[str((folder/'report.json').resolve())]=closed['report_sha256']
        predictor=read(args.predictor/'predictor.json');closed=read(args.predictor/'complete.json')
        assert sha(args.predictor/'predictor.json')==closed['predictor_sha256']
        pins[str((args.predictor/'predictor.json').resolve())]=closed['predictor_sha256']
        proof=read(args.geometry_proof/'report.json')
        tolerance=.005 if predictor.get('motor') else .001
        assert proof['geometry_version']==predictor['geometry_version'] and proof['max_error_fp32']<tolerance and proof['max_error_fp32_continuous']<.0002 and not proof['go_neural_execution']
        assert predictor['kind']=='combat_geometry_shot_predictor_v1' and predictor['input_width']==214
        model['geometry_aim_ranker']=dict(version='combat_geometry_aim_ranker_v1',offset_degrees=[0,-5,-2.5,2.5,5],predictor=predictor)
    else:
        assert args.predictor is None and args.geometry_proof is None
        if args.mode=='geometry_aim_explore_v1':
            actor=branch(source,'actor')
            assert actor.conditional_fire_branch is not None
            assert bool((actor.conditional_fire_branch[-1].weight==0).all()) and bool((actor.conditional_fire_branch[-1].bias==0).all()), 'Acquisition requires action-independent current-frame fire'
            model['geometry_aim_exploration']=dict(version='combat_geometry_aim_exploration_v1',offset_degrees=[0,-5,-2.5,2.5,5],baseline_probability=.5,sampler='splitmix64_frame_v1')
    restored=copy.deepcopy(model);restored.pop('execution_mode');restored.pop('geometry_aim_ranker',None);restored.pop('geometry_aim_exploration',None);assert restored==source
    try:validate_precision_model(model)
    except ValueError as exc:assert 'Inference-only execution mode' in str(exc)
    else:raise AssertionError('Standard PPO accepted inference-only execution')
    record=dict(version='combat_inference_execution_fork_v1',mode=args.mode,
        source_checkpoint_sha256=seal['checkpoint_sha256'],source_weights_sha256=seal['weights_sha256'],
        changed_fields=[k for k in model if k not in source or model[k]!=source[k]],standard_ppo_eligible=False)
    # CPU deserialization preserves CPU RNG storage; every numerical comparison is CUDA.
    fork=torch.load(args.source/'checkpoint.pt',map_location='cpu',weights_only=True)
    fork['inference_only_execution']=record
    args.out.mkdir(parents=True)
    durable_json(args.out/'weights.json',model)
    fork['weights_sha256']=sha(args.out/'weights.json')
    durable_write(args.out/'checkpoint.pt',lambda f:torch.save(fork,f))
    loaded=torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    for key in cp:
        if key!='weights_sha256':assert equal_tree(cp[key],loaded[key]),key
    assert loaded['inference_only_execution']==record
    receipt=dict(state='complete',version=record['version'],device='cuda',execution_mode=args.mode,
        learned_tensors_std_adam_rng_history_unchanged=True,weapon_pose_and_gaussian_sampling_unchanged=True,
        standard_ppo_rejected=True,source=str(args.source.resolve()),inputs=pins,script_sha256=sha(__file__),
        training_performed=False,promotion=False,
        scope=('Inference only. Learned MG critic ranks25nearby aim candidates; source stochastic target/mode/weapon/fire/movement retained, same RNG. No firing mask. Standard likelihood is counterfactual; PPO replay forbidden.' if args.predictor else
            'Inference only. Target/aim-mode argmax; optional conditional-fire logit sign. Consume legacy uniforms. No rule controller, target lock, firing mask, or weapon sampling change. Logged standard-policy likelihood is counterfactual and ineligible for PPO replay.'))
    if args.mode=='geometry_aim_explore_v1':
        receipt.update(purpose='predictive_train_acquisition_only',current_frame_fire_independent_of_intervention=True,
            scope='MG aim exploration: 50% baseline, 50% uniform24nonzero offsets; independent frame-hash stream. Source actor/value/std/Adam/RNG intact; target/mode/move/pose/weapon/fire sampling retained. Predictive-only TRAIN labels; standard PPO replay forbidden. Not a production policy candidate.')
    for path,digest in pins.items():assert sha(Path(path))==digest
    durable_json(args.out/'report.json',receipt)
    durable_json(args.out/'complete.json',dict(version='combat_update_complete_v1',weights_sha256=sha(args.out/'weights.json'),checkpoint_sha256=sha(args.out/'checkpoint.pt'),report_sha256=sha(args.out/'report.json')))
    print(args.mode,'sealed on CUDA; no training')


if __name__=='__main__':main()
