"""Inventory durable seed metadata without loading large traces or model weights."""
import argparse,json,pathlib,subprocess,hashlib
from process_combat_architecture_pool import read,save,sha

def collect(value,seeds):
    if isinstance(value,dict):
        for key,item in value.items():
            if key in ('seed','engine_seed') and isinstance(item,int):seeds.add(item)
            elif key=='seeds' and isinstance(item,list):seeds.update(v for v in item if isinstance(v,int))
            else:collect(item,seeds)
    elif isinstance(value,list):
        for item in value:collect(item,seeds)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evaluation',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[1];evaluation=a.evaluation.resolve();assert not a.out.exists();a.out.mkdir(parents=True)
    protocol=read(evaluation/'protocol.json');reference=read(protocol['evaluations'][0]['plan']);families=reference['tasks'];assert len(families)==20
    tests={t['episode']['id']:t['episode']['splits']['test'] for t in families};test_universe={s for v in tests.values() for s in range(v['start'],v['start']+min(v['count'],64))}
    # The live cohort is excluded from filesystem scanning, but all its frozen
    # plans must be development validation and disjoint from the reserved seeds.
    for entry in protocol['evaluations']:
        assert sha(entry['plan'])==entry['plan_sha256'];plan=read(entry['plan'])
        assert all(t['split']=='validation' for t in plan['tasks'])
        assert not {s for t in plan['tasks'] for s in t['seeds']} & test_universe
    paths=subprocess.check_output(['rg','--files','--no-ignore',str(repo/'workspace/artifacts'),'-g','*-result.json','-g','report.json','-g','manifest.json'],text=True).splitlines()
    seen=set();evidence=[];unreadable=[];scanned=0;excluded=0
    for filename in paths:
        path=pathlib.Path(filename)
        if path.is_relative_to(evaluation):excluded+=1;continue
        try:
            content=path.read_bytes();obj=json.loads(content.decode('utf-8-sig'))
        except (OSError,ValueError) as error:
            unreadable.append(dict(path=str(path),error=str(error)));continue
        local=set();collect(obj,local);seen.update(local);scanned+=1
        hits=sorted(local&test_universe)
        if hits:evidence.append(dict(path=str(path),sha256=hashlib.sha256(content).hexdigest(),seeds=hits))
    assert not unreadable,('Unverifiable historical metadata',unreadable[:5])
    selected=None
    for offset in range(57):
        wanted={v['start']+i for v in tests.values() for i in range(offset,offset+8)}
        if not wanted&seen:selected=offset;break
    assert selected is not None,'No common unseen eight-seed block in first64 test conditions'
    conditions=[dict(episode=key,seeds=list(range(v['start']+selected,v['start']+selected+8))) for key,v in tests.items()]
    save(a.out/'seed-inventory.json',dict(state='audited',metadata_files=scanned,active_metadata_files_excluded=excluded,unique_recorded_seeds=len(seen),recorded_seed_set_sha256=hashlib.sha256(json.dumps(sorted(seen)).encode()).hexdigest(),prior_test_seed_evidence=evidence,selected_test_offset=selected,episodes_per_arm=160,conditions=conditions,evaluation_protocol_sha256=sha(evaluation/'protocol.json'),scope='All current artifact pool-result/report/manifest JSON seed metadata, excluding the active cohort after checking every frozen validation plan. Excludes trace JSONL and weights. Unseen means no recorded use in this workspace metadata snapshot; no claim about external/imported unreceipted data. Plans alone are not counted as executed use.'))
    save(a.out/'selection-protocol.json',dict(version='combat_independent_test_selection_v1',state='prepared_not_dispatched',development_evaluation=str(evaluation),development_protocol_sha256=sha(evaluation/'protocol.json'),seed_inventory_sha256=sha(a.out/'seed-inventory.json'),split='test',seed_offset=selected,count_per_family=8,families=list(tests),episodes_per_arm=160,arms=['selected_candidate','parent3_before','firebc','rules'],total_episodes=640,slots=16,timescale=2,selection_order=['most development wins','fewest deaths','least mean received health damage','lexical candidate id'],eligibility='Candidate must exceed rules development wins, or tie wins with fewer deaths. Otherwise leave test reserved and improve using train/validation only.',scope='Choose exactly one candidate once from sealed full-registry development results. Freeze weights and all plans before test. Do not train or select another candidate on test results. Independent registered geometry distributions; no whole-map acceptance claim.'))
    print(json.dumps(dict(scanned=scanned,prior_test_metadata_files=len(evidence),reserved_offset=selected,episodes_per_arm=160)),flush=True)

if __name__=='__main__':main()
