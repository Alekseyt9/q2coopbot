"""CUDA audit and paired native ablation of sealed aim/movement BC branches."""
import argparse,copy,json,pathlib,shutil
from process_combat_architecture_pool import read,sha,save
from ppo_recurrent import torch,CausalAttention


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--template',type=pathlib.Path,required=True)
    ap.add_argument('--corpus',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True)
    a=ap.parse_args();repo=pathlib.Path(__file__).resolve().parents[1];out=a.out.resolve()
    template=read(a.template/'protocol.json');quality=read(a.template/'quality-report.json')
    assert quality['state']=='complete' and quality['protocol_sha256']==sha(a.template/'protocol.json')
    assert torch.cuda.is_available() and not out.exists()
    controls={e['model']:e for e in template['evaluations'] if e['model'] in ('firebc','rules')}
    assert set(controls)=={'firebc','rules'}
    baseline=read(controls['firebc']['plan']);source=pathlib.Path(baseline['model_path'])
    assert sha(source)==controls['firebc']['deterministic_weights_sha256']
    original=read(source);parent_sha=controls['firebc']['source_weights_sha256']
    entries=[('firebc','baseline',source,None),('joint','candidate',a.corpus/'joint-head-update-v1/weights.json',0.),
             ('coupling1','candidate',a.corpus/'coupled-head-update-v1/weights.json',1.),
             ('coupling16','candidate',a.corpus/'coupled-strong-update-v1/weights.json',16.),
             ('rules','baseline',None,None)]
    configs={0.:repo/'scripts/scenarios/combat-sequence-query-coordinated-v1.json',
             1.:repo/'scripts/scenarios/combat-sequence-query-coupled-v1.json',16.:a.corpus/'coupled-strong-config.json'}
    common_config=None
    audits=[];frozen=[];common=None
    for model,label,weight,coupling in entries:
        if coupling is None:continue
        folder=weight.parent;seal=read(folder/'complete.json');report=read(folder/'report.json')
        for filename,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
            assert sha(folder/filename)==seal[key]
        assert report['device']=='cuda' and report['parent']==parent_sha
        assert report['train_scope']=='aim_movement_heads' and report.get('world_input_weight',0)==coupling and report['epochs']==150
        assert sha(configs[coupling])==report['config_sha256']
        config=read(configs[coupling]);assert config.pop('world_input_weight',0)==coupling
        if common_config is None:common_config=config
        assert config==common_config,'BC configs differ beyond coupling weight'
        assert report['selected_action_rows']==[0,1,2,3] and report['unselected_parameters_exactly_preserved']
        conditions=(report['data_sha256'],report['train_context'],report['validation_context'],report['train_aim_rows'],report['validation_aim_rows'])
        if common is None:common=conditions
        assert conditions==common
        value=read(weight);spec=value['attention'];cp=torch.load(folder/'checkpoint.pt',map_location='cuda',weights_only=True)
        assert cp['weights_sha256']==seal['weights_sha256']
        assert cp['updates_completed']==report['ppo_updates_completed'] and cp['total_actor_steps']==report['ppo_total_actor_steps']
        assert cp['actor_optimizer']['state']=={},'BC branch should declare reset actor optimizer'
        for name in ('actor','value'):
            module=CausalAttention(value[name],spec[name],spec['heads'],spec['window']).to('cuda')
            reference=CausalAttention(original[name],original['attention'][name],original['attention']['heads'],original['attention']['window']).to('cuda')
            assert module.state_dict().keys()==cp[name].keys()==reference.state_dict().keys()
            for key,tensor in module.state_dict().items():
                assert torch.equal(tensor,cp[name][key]) and torch.isfinite(tensor).all()
                if name=='actor' and key.startswith(('head.','residual.')):
                    assert torch.equal(tensor[4:],reference.state_dict()[key][4:])
                else:assert torch.equal(tensor,reference.state_dict()[key])
            del module,reference
        assert torch.equal(torch.tensor(value['log_std'],device='cuda'),cp['log_std'])
        assert torch.equal(torch.tensor(value['log_std'],device='cuda'),torch.tensor(original['log_std'],device='cuda'))
        audits.append(dict(model=model,coupling_weight=coupling,weights_sha256=seal['weights_sha256'],
                           checkpoint_sha256=seal['checkpoint_sha256'],report_sha256=seal['report_sha256'],
                           inherited_ppo_updates=cp['updates_completed'],actor_optimizer='reset_by_BC',
                           actor_critic_std_match_checkpoint=True,unselected_parameters_match_FireBC=True))
        frozen.append((folder,seal));del cp
    out.mkdir();evaluations=[];plans=[];scenes=None
    for model,label,source,coupling in entries:
        control=controls['rules' if source is None else 'firebc'];plan=copy.deepcopy(read(control['plan']))
        assert sha(control['plan'])==control['plan_sha256']
        assert sha(plan['registry_path'])==plan['registry_sha256']
        folder=out/(model+'-'+label);folder.mkdir();deterministic_sha=None
        if source:
            weights=read(source);weights['deterministic']=True;save(folder/'weights.json',weights)
            deterministic_sha=sha(folder/'weights.json');plan['model_path']=str(folder/'weights.json');plan['model_sha256']=deterministic_sha
        plan['output_root']=str(folder/'capture')
        for task in plan['tasks']:assert sha(task['runner_path'])==task['runner_sha256']
        conditions=[{k:t[k] for k in ('episode','episode_sha256','split','seeds','instances','reward_sha256')} for t in plan['tasks']]
        if scenes is None:scenes=conditions
        assert scenes==conditions
        path=folder/'plan.json';save(path,plan);plans.append(str(path))
        evaluations.append(dict(model=model,label=label,root=plan['output_root'],plan=str(path),plan_sha256=sha(path),
             source_weights_sha256=sha(source) if source else None,deterministic_weights_sha256=deterministic_sha))
    save(out/'cuda-audit.json',dict(device='cuda',gpu=torch.cuda.get_device_name(0),checks=audits,scope='Serialized equality and preservation on CUDA; no new training, no live acceptance.'))
    save(out/'plans.json',plans)
    save(out/'protocol.json',dict(version='combat_coupling_paired_validation_v1',comparison_stage='aim_movement_coupling_ablation',
         evaluations=evaluations,families=template['families'],episodes_per_model=template['episodes_per_model'],
         total_episodes=5*template['episodes_per_model'],seed_offset=template['seed_offset'],slots=16,timescale=2,
         split='validation',final_test_deferred=True,validation_reused_for_tuning=True,
         comparison_reference='firebc-baseline',template_protocol_sha256=sha(a.template/'protocol.json'),
         template_quality_sha256=sha(a.template/'quality-report.json'),cuda_audit_sha256=sha(out/'cuda-audit.json'),
         scope='Same FireBC parent and 150 CUDA BC epochs on identical data; joint loss vs world-input coupling weights0/1/16. Paired native validation,20 families, reused conditions; no final-test superiority.'))
    save(out/'progress.json',dict(stage='plans_prepared',episodes=5*template['episodes_per_model']))
    print(json.dumps(dict(root=str(out),episodes=5*template['episodes_per_model'],cuda_audit='passed')))


if __name__=='__main__':main()
