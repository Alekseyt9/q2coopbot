"""Join complete native/CUDA counterfactual runs without emitting hard targets."""
import argparse
import copy
import json
from pathlib import Path

from combat_continuation_evidence import VERSION,INDEX_VERSION,summary,validate_record,load_manifest
from combat_continuation_probe import VERSION as FORK_VERSION
from combat_native_teacher_registry import load_index
from combat_wide_aim import PROBE_VERSION
from process_combat_architecture_pool import read,save,sha


def build(repo,index,source,roots,out):
    assert not out.exists()
    source=source.resolve();source_model=read(source);source_sha=sha(source)
    legacy={r['id']:r for r in load_index(repo,index)}
    pins={str(source):source_sha,str(index.resolve()):sha(index)}
    for name in ('build_combat_continuation_evidence_v2.py','combat_continuation_evidence.py','combat_native_teacher_registry.py'):
        path=repo/'scripts'/name;pins[str(path)]=sha(path)
    groups={}
    def pin(path,digest=None):
        path=Path(path).resolve();actual=sha(path)
        if digest is not None:assert actual==digest,str(path)
        if str(path) in pins:assert pins[str(path)]==actual
        pins[str(path)]=actual
    for corpus in read(index)['corpora']:
        path=repo/corpus['manifest'];pin(path,corpus['manifest_sha256'])
        for entry in read(path)['entries']:pin(entry['path'],entry['sha256'])
    for root in roots:
        root=root.resolve();completion=read(root/'complete.json');results=read(root/'results.json')
        assert completion['state']==results['state']=='complete'
        pin(root/'results.json',completion['results_sha256']);pin(root/'complete.json');pin(root/'exports.json')
        exports={(e['context_id'],e['future'],e['mode']):e for e in read(root/'exports.json')['entries']}
        for case in results['cases']:
            context=case['context'];record=legacy[context['record_id']]
            assert record['source_model_sha256']==source_sha
            assert all(record[k]==context[k] for k in ('map','seed','frame'))
            # The first pilot predates this report flag. The actual root
            # features/source latent are checked independently below for all arms.
            if 'historical_root_exact' in case:assert case['historical_root_exact']
            assert all(c['root_and_prefix_exact'] and c['matching_clamp_life_terminal_exact'] for c in case['checks'])
            group=groups.setdefault(record['id'],dict(record=record,streams={},contract=None))
            for difference in case['differences']:
                future=difference['future'];arms={};suffixes=[]
                for mode in ('choice0','choice1'):
                    entry=exports[(context['id'],future,mode)]
                    outcome=next(o for o in case['outcomes'] if o['future']==future and o['mode']==mode)
                    assert outcome['data']==entry['data']
                    directory=Path(entry['data']);model_path=Path(entry['model'])
                    pin(model_path,entry['model_sha256']);model=read(model_path)
                    clamp=model.pop('diagnostic_wide_gate_probe');fork=model.pop('diagnostic_continuation_probe',None)
                    assert clamp==dict(version=PROBE_VERSION,frame=record['frame'],choice=int(mode[-1])) and model==source_model
                    seed=entry['future_seed'];suffixes.append(seed)
                    if seed is None:assert fork is None
                    else:assert fork==dict(version=FORK_VERSION,fork_frame=record['frame']+1,seed=seed)
                    meta=read(directory/'report.json');proof=read(directory/'cuda-verification.json')
                    assert meta['diagnostic_only'] and not meta['ppo_training_ready']
                    assert meta['aim_gamma']==.99 and meta['feature_version']==record['input']['feature_version']
                    contract={k:meta[k] for k in ('reward_version','reward_config_sha256','combat_execution_contract')}
                    if group['contract'] is None:group['contract']=contract
                    else:assert group['contract']==contract,'Mixed reward or execution contracts'
                    assert proof['state']=='passed' and proof['device']=='cuda' and proof['model_sha256']==entry['model_sha256']
                    observed=None if seed is None else proof['diagnostic_continuation']['observed']
                    if seed is not None:assert proof['diagnostic_continuation']['fire_verified']
                    pin(directory/'report.json');pin(directory/'cuda-verification.json',meta['cuda_verification_sha256'])
                    pin(directory/'rollout.jsonl',meta['rollout_sha256'])
                    with (directory/'rollout.jsonl').open(encoding='utf-8-sig') as stream:
                        rows=[r for r in map(json.loads,stream) if r['frame']>=record['frame']]
                    assert rows and rows[0]['frame']==record['frame']
                    assert rows[0]['features']==record['input']['features']
                    assert rows[0]['sample']['wide_aim']['source_latent']==record['input']['source_latent']
                    utility=sum(.99**(r['frame']-record['frame'])*r['reward'] for r in rows)
                    assert abs(utility-outcome['utility'])<1e-6
                    arms[mode]=dict(utility=utility,rules_combat_rows=outcome['ownership']['rules_combat_rows'],
                        goal=outcome['goal'],death=outcome['death'],fork_observed=observed,data=str(directory))
                assert suffixes[0]==suffixes[1]
                assert abs(arms['choice1']['utility']-arms['choice0']['utility']-difference['delta'])<1e-6
                item=dict(suffix_seed=suffixes[0],fork_frame=None if suffixes[0] is None else record['frame']+1,outcomes=arms)
                if suffixes[0] in group['streams']:assert group['streams'][suffixes[0]]==item,'Reused stream changed'
                else:group['streams'][suffixes[0]]=item
    assert len(groups)==28,'Expected all twenty prior and all eight frozen confirmation contexts'
    out.mkdir(parents=True);entries=[]
    for rid,group in sorted(groups.items()):
        legacy_record=group['record'];streams=list(group['streams'].values())
        record=dict(version=VERSION,id=rid,source_model_sha256=source_sha,
            discount=.99,**group['contract'],
            **{k:legacy_record[k] for k in ('map','seed','frame')},input=copy.deepcopy(legacy_record['input']),
            streams=streams,summary=summary(streams),diagnostic_only=True,ordinary_ppo_eligible=False,training_target_emitted=False)
        validate_record(record);path=out/(rid+'.json');save(path,record)
        entries.append(dict(id=rid,map=record['map'],seed=record['seed'],path=str(path.resolve()),sha256=sha(path)))
    for path,digest in pins.items():assert sha(path)==digest,path
    save(out/'manifest.json',dict(version=VERSION,state='complete',entries=entries,contexts=len(entries),
        source_model_sha256=source_sha,inputs=pins,ordinary_ppo_eligible=False,training_targets_emitted=False,
        scope='Empirical paired suffix continuations on historical TRAIN roots,not exact Q or independent world replication. Causal inputs unchanged; map-level splits; no hard labels or actor fitting.'))
    assert len(load_manifest(out/'manifest.json'))==28
    save(out/'index.json',dict(version=INDEX_VERSION,ordinary_ppo_eligible=False,training_targets_emitted=False,split_unit='map',
        corpora=[dict(manifest=str((out/'manifest.json').relative_to(repo)),manifest_sha256=sha(out/'manifest.json'))]))
    print(out/'index.json',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('index','source','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--root',type=Path,action='append',required=True)
    a=ap.parse_args();repo=Path(__file__).resolve().parents[1];build(repo,a.index,a.source,a.root,a.out.resolve())
