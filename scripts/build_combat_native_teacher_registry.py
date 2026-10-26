"""Build a typed paired-native teacher corpus, explicitly unusable as PPO data."""
import argparse,copy,hashlib,json
from pathlib import Path
from ppo_combat import torch,training_devices
from combat_wide_aim import geometry,PROBE_VERSION
from combat_conditional_fire import inputs as fire_inputs
from process_combat_architecture_pool import read,save,sha

VERSION='combat_native_coupled_gate_teacher_v1'

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();source=a.source.resolve();out=a.out.resolve();assert not out.exists()
    training_devices();torch.set_num_threads(2)
    seal=read(source/'complete.json');assert seal['state']=='complete' and seal['diagnostic_only'] and not seal['training_performed']
    assert sha(source/'results.json')==seal['results_sha256'] and sha(source/'exports.json')==seal['exports_sha256']
    results=read(source/'results.json');exports=read(source/'exports.json')['entries'];protocol=read(source/'protocol.json')
    pins={str(p.resolve()):sha(p) for p in (Path(__file__),source/'complete.json',source/'results.json',source/'exports.json',source/'protocol.json')}
    out.mkdir(parents=True)
    save(out/'protocol.json',dict(version=VERSION,state='frozen',gamma=.99,return_tie_margin=.05,
        utility='Sum of observed shaped reward after root discounted by .99 per native frame; no critic bootstrap. Preference is finite observed-horizon utility, not exact infinite-horizon Q. Native kills/goals/deaths/damage retained separately.',
        eligibility='Both branches must be native/CUDA verified, same initial fixture and continuation weights, exact prefix/current observation, no rules-owned combat rows through terminal goal/death/horizon. Keep all cases including ties and ineligible rows.',
        inputs='Only root current observation features and original pre-wide frozen-actor event, selected target/mode/vertical/weapon and current recoil/anchor. No branch result or current wide choice/fire as input.',
        scope='Paired native teacher corpus; record count and source collection determine scope. This build does not fit or validate a policy.',
        diagnostic_only=True,ordinary_ppo_eligible=False,training_performed=False,promotion=False,inputs_sha256=pins))
    entries=[];counts={'choice0':0,'choice1':0,'tie':0,'ineligible':0}
    for case in results['cases']:
        assert all(case[k] for k in ('prefix_exact','current_observation_exact','matching_clamp_first_life_exact'))
        seed,frame=case['seed'],case['frame'];records={};model_core=None;reward_sha=None
        for label in ('baseline','choice0','choice1'):
            e=next(v for v in exports if v['seed']==seed and v['mode']==label);directory=Path(e['data']);model=read(e['model']);assert sha(e['model'])==e['model_sha256']
            core=copy.deepcopy(model);probe=core.pop('diagnostic_wide_gate_probe',None)
            if label=='baseline':assert probe is None;model_core=core
            else:assert core==model_core and probe==dict(version=PROBE_VERSION,frame=frame,choice=int(label[-1]))
            meta=read(directory/'report.json');proof=read(directory/'cuda-verification.json')
            assert proof['state']=='passed' and proof['device']=='cuda' and proof['model_sha256']==e['model_sha256']
            assert sha(directory/'rollout.jsonl')==meta['rollout_sha256'] and sha(directory/'cuda-verification.json')==meta['cuda_verification_sha256']
            if label!='baseline':
                assert meta['diagnostic_only'] and not meta['ordinary_training_eligible'] and not meta['ppo_training_ready']
                assert proof['wide_gate_clamp_and_final_fire_cdf_verified']
            reward_sha=meta['reward_config_sha256'] if reward_sha is None else reward_sha
            assert meta['reward_config_sha256']==reward_sha
            rows=[json.loads(l) for l in (directory/'rollout.jsonl').open(encoding='utf-8-sig')]
            root=next(r for r in rows if r['frame']==frame);rr=[r for r in rows if r['frame']>=frame]
            assert rr and all(r['seed']==seed for r in rr)
            utility=sum((.99**(r['frame']-frame))*r['reward'] for r in rr)
            records[label]=dict(root=root,utility=utility,rows=len(rr),ends_terminal=rr[-1]['terminal'],ends_truncated=rr[-1]['truncated'],data=str(directory),model_sha256=e['model_sha256'])
            for p in (Path(e['model']),directory/'report.json',directory/'cuda-verification.json',directory/'rollout.jsonl'):pins[str(p)]=sha(p)
        r0,r1=(records[k]['root'] for k in ('choice0','choice1'))
        assert r0['features']==r1['features']==records['baseline']['root']['features']
        for k in ('source_latent','anchor_degrees','fire_uniform','uniform'):
            assert r0['sample']['wide_aim'][k]==r1['sample']['wide_aim'][k]
        assert all(r0['sample'].get(k,0)==r1['sample'].get(k,0) for k in ('target','aim_mode','vertical','weapon'))
        x=torch.tensor([r0['features']],device='cuda',dtype=torch.float64)
        cat=lambda k:torch.tensor([r0['sample'].get(k,0)],device='cuda',dtype=torch.long)
        target,mode,vertical,weapon=(cat(k) for k in ('target','aim_mode','vertical','weapon'))
        latent=torch.tensor([r0['sample']['wide_aim']['source_latent']],device='cuda',dtype=torch.float64)
        anchor,active=geometry(x,target,mode);assert bool(active[0])
        ctx=torch.cat((fire_inputs(x,target,mode,latent,vertical,weapon),x[:,810:814],anchor/180),-1)
        assert ctx.shape==(1,187) and bool(torch.isfinite(ctx).all())
        reasons=[]
        for label in ('choice0','choice1'):
            if case['outcomes'][label]['rules_combat_rows']:reasons.append(label+':rules_owned_continuation')
        delta=records['choice1']['utility']-records['choice0']['utility']
        preference=None if abs(delta)<.05 else int(delta>0)
        if reasons:counts['ineligible']+=1
        elif preference is None:counts['tie']+=1
        else:counts['choice'+str(preference)]+=1
        condition_plan=next(Path(p) for p in protocol['inputs'] if p.endswith(f'{seed}-baseline-plan.json'))
        fixture=read(condition_plan)['tasks'][0]['instances'][0]
        parent=records['baseline']['model_sha256'];identity=hashlib.sha256(json.dumps([fixture,parent,frame],sort_keys=True).encode()).hexdigest()
        entry=dict(version=VERSION,id=identity,map=case['map'],seed=seed,frame=frame,source_model_sha256=parent,fixture=fixture,
            input=dict(feature_version=model_core['feature_version'],features=r0['features'],pre_gate_context187=ctx[0].cpu().tolist(),source_latent=latent[0].cpu().tolist(),target=int(target[0]),aim_mode=int(mode[0]),vertical=int(vertical[0]),weapon=int(weapon[0])),
            outcomes={k:dict(native=case['outcomes'][k],discounted_observed_return=records[k]['utility'],observed_return_rows=records[k]['rows'],ends_terminal=records[k]['ends_terminal'],ends_truncated=records[k]['ends_truncated'],data=records[k]['data']) for k in ('choice0','choice1')},
            target=dict(preference=preference if not reasons else None,utility_delta=delta,tie_margin=.05,eligible_for_teacher_supervision=not reasons,ineligible_reasons=reasons),
            coupled_fire_verified=True,same_continuation_policy=True,ordinary_ppo_eligible=False,diagnostic_only=True,training_performed=False)
        path=out/(identity+'.json');save(path,entry);entries.append(dict(id=identity,map=case['map'],seed=seed,path=str(path),sha256=sha(path),preference=entry['target']['preference'],teacher_eligible=not reasons))
    for p,h in pins.items():assert sha(p)==h
    save(out/'manifest.json',dict(version=VERSION,state='complete',entries=entries,counts=counts,inputs=pins,ordinary_ppo_eligible=False,diagnostic_only=True,training_performed=False,promotion=False))
    save(out/'complete.json',dict(state='complete',manifest_sha256=sha(out/'manifest.json'),records=len(entries),training_performed=False,ordinary_ppo_eligible=False,promotion=False))
    print(json.dumps(dict(records=len(entries),counts=counts)))

if __name__=='__main__':main()
