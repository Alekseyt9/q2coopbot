"""Verify behavior likelihood/context and finalize exact native bootstrap on CUDA."""
import argparse
import json
import pathlib
import shutil

from process_combat_architecture_pool import read, save, sha
from ppo_combat import network, torch
from ppo_recurrent import prepare_context, sequence, Recurrent
from combat_attention import CausalAttention, EntityAttention
from combat_cuda_bootstrap import next_values
from combat_sequence_storage import preserve_sequence


def finalize(model_path, data, out):
    assert torch.cuda.is_available(), 'CUDA is required; no CPU fallback'
    assert not out.exists(), 'Fresh CUDA output required'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    model, meta = read(model_path), read(data / 'report.json')
    assert meta['version'] == 'combat_ppo_native_cuda_pending_v1'
    assert meta['numerical_verification'] == 'cuda_pending'
    assert meta['model_sha256'] == sha(model_path) and not model['deterministic']
    assert meta['feature_version'] == model['feature_version']
    assert meta.get('action_distribution','') == model.get('action_distribution',''), 'Action distribution differs'
    joint,core_model=None,model
    if model.get('action_distribution'):
        from combat_joint_aim_policy import JointAim,base_model
        diagnostic_probe=bool(model.get('diagnostic_joint_aim_probe') or model.get('diagnostic_movement_probe'))
        core_model=base_model(model,allow_probe=diagnostic_probe,allow_shared=True)
        joint=JointAim(model,allow_probe=diagnostic_probe,allow_shared=True)
    paired = meta.get('paired_adapter_version')
    if paired:
        assert paired == 'coop_primary_native_adapter_v1' and meta['paired_adapter_verified']
        assert meta['reward_version'] == 'combat_reward_v11'
        end = meta.get('paired_episode_end', 'joint_death')
        assert end in ('joint_death', 'surviving_horizon', 'joint_death_after_handoff')
        assert meta['terminals'] == (1 if end == 'joint_death' else 0)
        assert not meta['paired_training_ready']
    assert sha(data / 'native-rollout.jsonl') == meta['native_rollout_sha256']
    sources = dict(meta['source_sha256'])
    sources.update({str((data / 'report.json').resolve()): sha(data / 'report.json'),
                    str((data / 'native-rollout.jsonl').resolve()): meta['native_rollout_sha256'],
                    str(pathlib.Path(__file__).resolve()): sha(__file__),
                    str(pathlib.Path(__file__).with_name('combat_cuda_bootstrap.py').resolve()):
                        sha(pathlib.Path(__file__).with_name('combat_cuda_bootstrap.py')),
                    str(pathlib.Path(__file__).with_name('combat_sequence_storage.py').resolve()):
                        sha(pathlib.Path(__file__).with_name('combat_sequence_storage.py'))})
    weapon_helpers=[]
    if joint is not None:
        if model.get('diagnostic_movement_probe'):
            weapon_helpers.append(pathlib.Path(__file__).with_name('combat_movement_probe.py').resolve())
        weapon_helpers.extend(pathlib.Path(__file__).with_name(name).resolve() for name in
            ('combat_joint_aim_policy.py','combat_aim_candidates.py','combat_action_effect.py','combat_action_geometry.py','combat_conditional_fire.py'))
    if model.get('conditional_fire'):
        weapon_helpers.append(pathlib.Path(__file__).with_name('combat_conditional_fire.py').resolve())
    if model.get('weapon_state_residual'):
        weapon_helpers.extend(pathlib.Path(__file__).with_name(name).resolve() for name in
            ('combat_weapon_state_residual.py','combat_spatial_aim.py','ppo_recurrent.py','ppo_combat.py',
             'combat_target_head.py','combat_precision_head.py','combat_weapon_head.py','combat_attention.py'))
    if model.get('weapon_exploration'):
        weapon_helpers.extend(pathlib.Path(__file__).with_name(name).resolve() for name in
                              ('combat_weapon_exploration.py','combat_weapon_head.py','combat_spatial_aim.py'))
    sources.update({str(path):sha(path) for path in weapon_helpers})
    if model.get('sampled_motion_aim'):
        helper=pathlib.Path(__file__).with_name('combat_sampled_motion_aim.py').resolve()
        sources[str(helper)]=sha(helper)
    for path, digest in sources.items():
        assert sha(path) == digest, 'Native source changed: ' + path
    rows = [json.loads(line) for line in (data / 'native-rollout.jsonl').read_text().splitlines()]
    assert len(rows) == meta['rows'] > 0
    assert all(r['sample']['version'] == meta['policy_version'] for r in rows)
    context, prepared = [], None
    if meta.get('recurrent_version'):
        assert sha(data / 'sequence.jsonl') == meta['sequence_sha256']
        context = [json.loads(line) for line in (data / 'sequence.jsonl').read_text().splitlines()]
        assert len(context) == meta['sequence_rows']
        sources[str((data / 'sequence.jsonl').resolve())] = meta['sequence_sha256']
        prepared = prepare_context(context, rows, 'cuda')

    def build(name):
        if model.get('memory'):
            module = Recurrent(model[name], model['memory'][name])
        elif model.get('attention'):
            spec = model['attention']
            module = CausalAttention(model[name], spec[name], spec['heads'], spec['window'])
        elif model.get('entity_attention'):
            spec = model['entity_attention']
            module = EntityAttention(model[name], spec[name], spec['heads'])
        else:
            module = network(model[name])
        if name == 'actor' and model.get('spatial_aim'):
            from combat_spatial_aim import wrap
            module = wrap(module, core_model)
        return module.to('cuda')

    actor, value = build('actor'), build('value')
    x = torch.tensor([r['features'] for r in rows], device='cuda', dtype=torch.float32)
    nx = torch.tensor([r['next_features'] for r in rows], device='cuda', dtype=torch.float32)
    assert x.shape == nx.shape and bool(torch.isfinite(x).all()) and bool(torch.isfinite(nx).all())
    reset = torch.tensor([r['next_reset'] for r in rows], device='cuda', dtype=torch.bool)
    zero = torch.tensor([r['bootstrap_zero'] for r in rows], device='cuda', dtype=torch.bool)
    assert all(not r['terminal'] or r['bootstrap_zero'] for r in rows)
    assert all(r['next_frame'] == r['frame'] + 1 or r['next_reset'] for r in rows)
    z = torch.tensor([r['sample']['latent'] for r in rows], device='cuda', dtype=torch.float32)
    attack = torch.tensor([float(r['sample']['attack']) for r in rows], device='cuda')
    vertical = torch.tensor([r['sample']['vertical'] for r in rows], device='cuda', dtype=torch.long)
    weapon = torch.tensor([r['sample'].get('weapon', 0) for r in rows], device='cuda', dtype=torch.long)
    target = torch.tensor([r['sample'].get('target', 0) for r in rows], device='cuda', dtype=torch.long)
    mode = torch.tensor([r['sample'].get('aim_mode', 0) for r in rows], device='cuda', dtype=torch.long)
    std = torch.tensor(model['log_std'], device='cuda', dtype=torch.float32)

    def evaluate(module, name):
        if prepared is not None:
            return sequence(module, prepared, 32, True, name)
        if model.get('entity_attention'):
            return module.single(x), 0.
        return module(x), 0.

    with torch.no_grad():
        raw, actor_memory_error = evaluate(actor, 'actor')
        predicted, critic_memory_error = evaluate(value, 'value')
        candidate_error=None; movement_draw_error=None
        if joint is not None:
            from combat_joint_aim_policy import events,probabilities
            from combat_aim_candidates import log_probability,frame_uniforms,choose
            source,choice,timing,active=events(rows,x,target,mode,vertical,weapon)
            score_inputs=None
            if joint.geometry_head:
                # The live Go scorer uses float64. Preserve original JSON
                # inputs for geometry differences and hash-CDF verification;
                # training and prototype export remain float32 on CUDA.
                joint.double()
                sf=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float64)
                sz=torch.tensor([(r['sample'].get('aim_candidate') or {}).get('source_latent',r['sample']['latent']) for r in rows],device='cuda',dtype=torch.float64)
                st=torch.tensor([(r['sample'].get('aim_candidate') or {}).get('timing',[0.,0.]) for r in rows],device='cuda',dtype=torch.float64)
                score_inputs=(sf,sz,st)
            frame_tensor=torch.tensor([r['frame'] for r in rows],device='cuda',dtype=torch.long)
            movement_shift=None
            if model.get('diagnostic_movement_probe'):
                from combat_movement_probe import shifts as movement_shifts
                movement_shift=movement_shifts(model,frame_tensor)
            lp,_,logits,score_active=probabilities(joint,raw,std,source,z,attack,vertical,weapon,x,target,mode,choice,timing,
                getattr(actor,'sampled_motion_aim_branch',None),getattr(actor,'motor_branch',None),getattr(actor,'conditional_fire_branch',None),
                frames=frame_tensor,score_inputs=score_inputs,movement_shift=movement_shift)
            assert torch.equal(active,score_active)
            if model.get('diagnostic_movement_probe'):
                from combat_movement_probe import verify_draw_receipts
                movement_draw_error=verify_draw_receipts(model,rows,raw,std,source,z,x,target,mode,vertical,weapon,actor)
            expected_category=torch.tensor([r['sample']['aim_candidate']['log_probability'] if r['sample'].get('aim_candidate') else 0. for r in rows],device='cuda')
            candidate_error=float((log_probability(logits,choice,active)-expected_category).abs().max())
            assert candidate_error<.003, candidate_error
            if model['aim_candidate_head'].get('shared_scorer'):
                records=[r['sample']['aim_candidate']['logits'] for r in rows if r['sample'].get('aim_candidate')]
                assert all(len(v)==25 and all(isinstance(q,(int,float)) for q in v) for v in records)
                recorded=torch.tensor(records,device='cuda',dtype=logits.dtype).reshape(-1,25)
                shared_logit_error=float((logits[active]-recorded).abs().max()) if records else 0.
                shared_probability_error=float((logits[active].softmax(-1)-recorded.softmax(-1)).abs().max()) if records else 0.
                assert bool(torch.isfinite(recorded).all()) and shared_logit_error<.001 and shared_probability_error<.0001,(shared_logit_error,shared_probability_error)
            # The native exporter accepts first-life rows only.
            uniforms=torch.tensor([frame_uniforms(r['sample']['sampling_seed'],r['frame'],1) for r in rows],device='cuda',dtype=torch.float64)
            assert torch.equal(choose(logits.double(),uniforms[:,0],uniforms[:,1],active),choice),'Frame-hash categorical assignment differs'
        elif model.get('aim_mode_head'):
            from combat_precision_head import probabilities
            lp, _ = probabilities(raw, std, z, attack, vertical, weapon, x, target, mode,
                                  getattr(actor, 'sampled_motion_aim_branch', None),getattr(actor,'motor_branch',None),getattr(actor,'conditional_fire_branch',None))
        elif model.get('target_head'):
            from combat_target_head import probabilities
            lp, _ = probabilities(raw, std, z, attack, vertical, weapon, x, target)
        else:
            from combat_weapon_head import probabilities, feature_mask
            lp, _ = probabilities(raw, std, z, attack, vertical,
                                   weapon if model.get('weapon_head') else None,
                                   feature_mask(x) if model.get('weapon_head') else None)
        log_error = float((lp - torch.tensor([r['sample']['log_probability'] for r in rows], device='cuda')).abs().max())
        value_error = float((predicted[:, 0] - torch.tensor([r['sample']['value'] for r in rows], device='cuda',dtype=predicted.dtype)).abs().max())
        initial_value_error=value_error
        critic_precision='float32'
        bootstrap_prepared=prepared
        bootstrap_features=nx
        if value_error>=1e-4:
            # Match the live Go float64 critic without weakening acceptance.
            # Recreate inputs from JSON; promoting already-rounded float32
            # context would retain the error being diagnosed.
            value=value.double()
            exact_x=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float64)
            bootstrap_features=torch.tensor([r['next_features'] for r in rows],device='cuda',dtype=torch.float64)
            bootstrap_prepared=prepare_context(context,rows,'cuda',dtype=torch.float64) if prepared is not None else None
            if bootstrap_prepared is not None:
                predicted,critic_memory_error=sequence(value,bootstrap_prepared,32,True,'value')
            elif model.get('entity_attention'):
                predicted=value.single(exact_x)
            else:predicted=value(exact_x)
            value_error=float((predicted[:,0]-torch.tensor([r['sample']['value'] for r in rows],device='cuda',dtype=torch.float64)).abs().max())
            critic_precision='float64_fallback'
        assert log_error < .003 and value_error < 1e-4 and max(actor_memory_error, critic_memory_error) < 2e-5, (
            log_error, value_error, actor_memory_error, critic_memory_error)
        bootstrap = next_values(value, bootstrap_features, bootstrap_prepared, reset=reset)
        bootstrap[zero] = 0
        assert bool(torch.isfinite(bootstrap).all())
        values = bootstrap.cpu().tolist()  # Serialization only; neural computation stays on CUDA.
    for path, digest in sources.items():
        assert sha(path) == digest, 'Input changed during CUDA verification: ' + path
    out.mkdir()
    # Preserve the verifier actually used; future source edits must not invalidate
    # already verified rollouts or require keeping a mutable checkout unchanged.
    frozen = out / 'verifier-sources'
    frozen.mkdir()
    helper_paths=[pathlib.Path(__file__).resolve(), pathlib.Path(__file__).with_name('combat_cuda_bootstrap.py').resolve(),
                  pathlib.Path(__file__).with_name('combat_sequence_storage.py').resolve()]
    helper_paths.extend(weapon_helpers)
    if model.get('shared_target'):
        helper_paths.extend(pathlib.Path(__file__).with_name(name).resolve() for name in
                            ('combat_shared_target.py','combat_spatial_aim.py'))
    for path in dict.fromkeys(helper_paths):
        target = frozen / path.name
        shutil.copy2(path, target)
        sources.pop(str(path), None)
        sources[str(target.resolve())] = sha(target)
    with (out / 'rollout.jsonl').open('w', encoding='utf-8') as stream:
        for row, nv in zip(rows, values):
            row['next_value'] = nv
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    if prepared is not None:
        preserve_sequence(data / 'sequence.jsonl', out / 'sequence.jsonl', meta['sequence_sha256'])
    audit = dict(state='passed', device='cuda', rows=len(rows), context_rows=len(context),
                 log_probability_max_error=log_error, value_max_error=value_error,
                 actor_memory_max_error=actor_memory_error, critic_memory_max_error=critic_memory_error,
                 critic_precision=critic_precision,initial_float32_value_max_error=initial_value_error,
                 next_resets=int(reset.sum()), zero_bootstraps=int(zero.sum()),
                 native_rollout_sha256=meta['native_rollout_sha256'], model_sha256=sha(model_path),
                 scope='Exact native Step.Next features and reset flags; numerical checks and bootstrap on CUDA. Go performs only native proof, feature extraction and sampled action contract checks.')
    if joint is not None:
        audit.update(action_distribution=model['action_distribution'],candidate_log_probability_max_error=candidate_error,
                     original_latent_transform_verified=True,frame_hash_candidate_assignments_verified=True)
    if model.get('diagnostic_joint_aim_probe'):
        audit.update(diagnostic_probe_version=model['diagnostic_joint_aim_probe']['version'],
                     diagnostic_probe_frame=model['diagnostic_joint_aim_probe']['frame'],
                     frame_conditioned_logits_verified=True,ordinary_training_eligible=False)
    if model.get('diagnostic_movement_probe'):
        audit.update(diagnostic_movement_version=model['diagnostic_movement_probe']['version'],
                     diagnostic_movement_frame=model['diagnostic_movement_probe']['frame'],
                     diagnostic_movement_shift=model['diagnostic_movement_probe']['mean_shift'],
                     diagnostic_gaussian_noise_max_error=movement_draw_error,diagnostic_fire_draw_verified=True,ordinary_training_eligible=False)
    if model.get('aim_candidate_head',{}).get('shared_scorer'):
        audit.update(shared_candidate_all_logits_max_error=shared_logit_error,shared_candidate_all_probabilities_max_error=shared_probability_error,
                     shared_candidate_frames=len(records),shared_candidate_precision='float64' if joint.geometry_head else 'float32',ordinary_training_eligible=False)
    save(out / 'cuda-verification.json', audit)
    sources[str((out / 'cuda-verification.json').resolve())] = sha(out / 'cuda-verification.json')
    meta.update(version='combat_ppo_rollout_v1', numerical_verification='cuda_verified_v1',
                rollout_sha256=sha(out / 'rollout.jsonl'), source_sha256=sources,
                cuda_verification_sha256=sha(out / 'cuda-verification.json'))
    if model.get('diagnostic_joint_aim_probe'):
        meta.update(version='combat_diagnostic_joint_aim_probe_rollout_v1',diagnostic_only=True,ppo_training_ready=False,
                    diagnostic_probe_version=model['diagnostic_joint_aim_probe']['version'])
    if model.get('diagnostic_movement_probe'):
        meta.update(version='combat_diagnostic_movement_probe_rollout_v1',diagnostic_only=True,ppo_training_ready=False,
                    diagnostic_movement_version=model['diagnostic_movement_probe']['version'])
    if model.get('aim_candidate_head',{}).get('shared_scorer'):
        meta.update(shared_scorer_training_supported=False,ppo_training_ready=False)
    if paired:
        meta['paired_training_ready'] = True
        meta['paired_training_scope'] = 'Experimental verified primary learner pilot; scripted peer excluded. Registry-wide eligibility unproven.'
    save(out / 'report.json', meta)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=pathlib.Path)
    parser.add_argument('--data', required=True, type=pathlib.Path)
    parser.add_argument('--out', required=True, type=pathlib.Path)
    args = parser.parse_args()
    print(json.dumps(finalize(args.model.resolve(), args.data.resolve(), args.out.resolve())), flush=True)


if __name__ == '__main__':
    main()
