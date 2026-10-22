"""CUDA checkpoint/common-history audit and cross-teacher error matrix."""
import argparse
import gzip
import json
import pathlib
from ppo_recurrent import torch, prepare_context, sequence
from migrate_combat_navigation_cuda import branch
from combat_precision_head import selected_means
from combat_sampled_motion_aim import correction, VERTICAL_CONTROL_VERSION
from fork_conditional_weapon_exploration_cuda import equal_tree
from process_combat_architecture_pool import read,save,sha


def main():
    ap=argparse.ArgumentParser()
    for name in ('protocol','upper','center','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    args=ap.parse_args();assert torch.cuda.is_available() and not args.out.exists()
    torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    pins={}
    def pin(path,expected=None):
        path=pathlib.Path(path);digest=sha(path);assert expected is None or digest==expected
        pins[str(path.resolve())]=digest;return path
    protocol=read(pin(args.protocol))
    assert protocol['device']=='cuda' and protocol['common_support_only'] and not protocol['selection_on_development']
    for item in protocol['inputs']:pin(item['path'],item['sha256'])
    source=pathlib.Path(protocol['source_update']);seal=read(pin(source/'complete.json'))
    parent=torch.load(pin(source/'checkpoint.pt',seal['checkpoint_sha256']),map_location='cuda',weights_only=True)
    models={};reports={};labels={};metas={}
    for name,path in [('upper',args.upper),('center',args.center)]:
        closed=read(pin(path/'complete.json'))
        for file,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:pin(path/file,closed[key])
        model=read(path/'weights.json');report=read(path/'report.json')
        assert model['sampled_motion_aim']['version']==VERTICAL_CONTROL_VERSION
        assert report['source_checkpoint_sha256']==seal['checkpoint_sha256'] and report['steps']==protocol['steps_per_variant']
        for key in ('learning_rate','retention_weight'):assert report[key]==protocol[key]
        assert report['device']=='cuda' and not report['selection_on_development'] and report['vertical_columns_norm']==0
        cp=torch.load(path/'checkpoint.pt',map_location='cuda',weights_only=True)
        assert cp['weights_sha256']==closed['weights_sha256']
        actor=branch(model,'actor');assert equal_tree(cp['actor'],actor.state_dict())
        assert equal_tree(cp['value'],parent['value']) and torch.equal(cp['log_std'],parent['log_std'])
        for key,value in cp['actor'].items():
            if not key.startswith('sampled_motion_aim_branch.'):assert torch.equal(value,parent['actor'][key])
        models[name]=actor;reports[name]=report
        meta=read(pin(pathlib.Path(protocol['labels'])/name/'report.json'));assert meta['body_point']==name and meta['common_support_only']
        metas[name]=meta;binding=meta['splits']['development']
        with gzip.open(pin(binding['output'],binding['sha256']),'rt',encoding='utf-8') as stream:labels[name]=list(map(json.loads,stream))
    assert reports['upper']['retention_train_keys']==reports['center']['retention_train_keys']
    for a,b in zip(labels['upper'],labels['center']):
        for key in ('seed','index','features','sample'):assert a[key]==b[key]
    assert len(labels['upper'])==len(labels['center'])==protocol['development_examples']
    root=pathlib.Path(metas['upper']['corpus'])/'processing-development/merged'
    assert metas['upper']['corpus']==metas['center']['corpus']
    report=read(pin(root/'report.json'));streams={}
    for name in ('rollout','sequence'):
        path=pin(root/(name+'.jsonl'),report[name+'_sha256'])
        streams[name]=[json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()]
    rows=streams['rollout'];lookup={(r['seed'],r['index']):i for i,r in enumerate(rows)};members=labels['upper']
    ix=torch.tensor([lookup[(r['seed'],r['index'])] for r in members],device='cuda',dtype=torch.long)
    x=torch.tensor([r['features'] for r in members],device='cuda',dtype=torch.float32)
    move=torch.tensor([r['sample']['latent'][:2] for r in members],device='cuda',dtype=torch.float32)
    target=torch.tensor([r['sample']['target'] for r in members],device='cuda',dtype=torch.long)
    mode=torch.tensor([r['sample'].get('aim_mode',0) for r in members],device='cuda',dtype=torch.long)
    vertical=torch.tensor([r['sample']['vertical'] for r in members],device='cuda',dtype=torch.long)
    matrix={};predictions={}
    with torch.no_grad():
        prepared=prepare_context(streams['sequence'],rows,'cuda');reference=None
        for name,actor in models.items():
            raw,error=sequence(actor,prepared,check_states=True);assert error<2e-5
            if reference is None:reference=raw
            else:torch.testing.assert_close(raw,reference,rtol=0,atol=1e-5)
            raw=raw[ix];scale=torch.where(mode==1,15.,180.)[:,None]
            means=selected_means(raw,target,mode)[:,2:4]+correction(actor.sampled_motion_aim_branch,x,raw,move,target,mode,vertical)
            predicted=means.tanh()*scale;predictions[name]=predicted;matrix[name]={}
            for teacher,teacher_rows in labels.items():
                desired=torch.tensor([r['offline_label']['desired_command_delta_degrees'][::-1] for r in teacher_rows],device='cuda',dtype=torch.float32)
                error=(predicted-desired).abs()
                matrix[name][teacher]=dict(mean_absolute_axis_error_degrees=float(error.mean()),yaw_error_degrees=float(error[:,0].mean()),pitch_error_degrees=float(error[:,1].mean()))
        change=predictions['center']-predictions['upper']
        shift=dict(mean_signed_pitch_change_degrees=float(change[:,1].mean()),mean_absolute_axis_change_degrees=float(change.abs().mean()),max_axis_change_degrees=float(change.abs().max()))
    for p,h in pins.items():assert sha(p)==h
    save(args.out,dict(state='complete',version='combat_bodypoint_pair_audit_v1',device='cuda',samples=len(members),
        common_examples=True,matched_initial_checkpoint=True,frozen_original_actor_value_std=True,
        error_matrix=matrix,mean_action_change=shift,inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),
        training_performed=False,promotion=False,
        scope='Rows are identical across teacher variants; error only comparable within one teacher column. Frozen raw actor and categorical/std/value parameters verified on CUDA. No bbox visibility guarantee, live hit probability, optimum or combat improvement claim.'))
    print(json.dumps(dict(error_matrix=matrix,mean_action_change=shift)))


if __name__=='__main__':main()
