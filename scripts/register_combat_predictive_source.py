"""Register a sealed predictive-only capture recipe; never approve PPO replay."""
import argparse
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture',type=Path,required=True);ap.add_argument('--registry',type=Path,required=True)
    args=ap.parse_args();root=args.capture.resolve();target=args.registry.resolve()
    seal=read(root/'complete.json');assert seal['state']=='complete' and not seal['ppo_ready'] and not seal['policy_training']
    recipe=read(root/'recipe.json');assert sha(root/'recipe.json')==seal['recipe_sha256']
    assert recipe['split']=='train' and recipe['purpose']=='shot_predictive_train_only'
    data_seal=read(root/'dataset/complete.json');assert sha(root/'dataset/complete.json')==seal['dataset_complete_sha256']
    report=read(root/'dataset/report.json');assert sha(root/'dataset/report.json')==data_seal['report_sha256']
    assert sha(root/'dataset/dataset.pt')==data_seal['dataset_sha256'] and not data_seal['ppo_ready']
    value=read(target) if target.exists() else dict(version='combat_predictive_source_registry_v1',purpose='predictive_train_only',entries=[])
    assert value['version']=='combat_predictive_source_registry_v1' and value['purpose']=='predictive_train_only'
    entry=dict(kind='randomized_aim_capture_v1',capture=str(root),complete_sha256=sha(root/'complete.json'),
        recipe=str(root/'recipe.json'),recipe_sha256=seal['recipe_sha256'],dataset=str(root/'dataset'),
        dataset_complete_sha256=seal['dataset_complete_sha256'],dataset_sha256=data_seal['dataset_sha256'],
        source_model_sha256=recipe['inputs'][str(Path(recipe['source'])/'weights.json')],
        behavior_sampler=recipe['behavior_sampler'],conditions_registry=recipe['registry'],episodes=recipe['episodes'],
        seeds=recipe['seeds'],count_per_episode=recipe['count'],native_cases=seal['native_cases'],shots=report['shots'],
        perturbed_shots=report['perturbed_shots'],split='train',ppo_ready=False)
    old=next((e for e in value['entries'] if e['capture']==str(root)),None)
    if old is not None:assert old==entry
    else:
        used={seed for e in value['entries'] for seed in e['seeds']}
        assert not used.intersection(entry['seeds']),'Repeated engine seeds across capture recipes'
        value['entries'].append(entry)
    value.update(native_cases=sum(e['native_cases'] for e in value['entries']),shots=sum(e['shots'] for e in value['entries']),
        perturbed_shots=sum(e['perturbed_shots'] for e in value['entries']),ppo_ready=False)
    target.parent.mkdir(parents=True,exist_ok=True);save(target,value)
    print('Predictive recipes:',len(value['entries']),'cases:',value['native_cases'],'shots:',value['shots'])


if __name__=='__main__':main()
