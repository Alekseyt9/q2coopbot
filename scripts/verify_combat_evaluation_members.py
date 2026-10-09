"""Recover evaluation proofs from completed jobs, without changing capture receipts."""
import argparse, hashlib, json, pathlib, subprocess
from rebind_combat_architecture_evaluation import read, sha, save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True)
    ap.add_argument('--source-fingerprint',required=True);ap.add_argument('--native-fingerprint',required=True);a=ap.parse_args()
    root=a.root.resolve();protocol=read(root/'protocol.json');jobs=[read(p) for p in (root/'pool/jobs').glob('*-result.json')]
    assert len(jobs)==protocol['total_episodes'] and not any(j['error'] for j in jobs)
    assert len({j['job'] for j in jobs})==len(jobs)
    by_key={(j['plan_index'],j['task_index'],j['seed'],j['mode']):j for j in jobs}; assert len(by_key)==len(jobs)
    repo=pathlib.Path(__file__).resolve().parents[1];source_ids=set();native_ids=set();members={};binary_examples={};hashcache={}
    def cached_sha(path):
        path=pathlib.Path(path);s=path.stat();key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        if key not in hashcache:
            with path.open('rb') as stream: hashcache[key]=hashlib.file_digest(stream,'sha256').hexdigest()
        return hashcache[key]
    reference=None
    for pi,entry in enumerate(protocol['evaluations']):
        assert sha(entry['plan'])==entry['plan_sha256'];plan=read(entry['plan'])
        assert sha(plan['registry_path'])==plan['registry_sha256']
        if plan.get('model_path'):assert sha(plan['model_path'])==entry['deterministic_weights_sha256']
        for ti,task in enumerate(plan['tasks']):
            mode=task['modes'][0]
            for seed in task['seeds']:
                job=by_key[(pi,ti,seed,mode)];folder=pathlib.Path(job['root'])
                assert folder==pathlib.Path(entry['root'])/f'case-{ti}-{mode}'/f's-{seed}'
                manifest=read(folder/'manifest.json');report=read(folder/'report.json')
                assert report['capture_complete'] and report['provenance_valid'] and len(report['results'])==1
                result=report['results'][0]
                assert result['seed']==seed and all(result[k] for k in ('seed_confirmed','capture_valid','dispatch_valid','frame_budget_valid'))
                source_ids.add(manifest['source_fingerprint']);native_ids.add(manifest['native_source_fingerprint'])
                assert manifest['source_fingerprint']==a.source_fingerprint
                assert manifest['native_source_fingerprint']==a.native_fingerprint
                if reference is None:
                    reference=manifest
                    for record in manifest['sources']:assert cached_sha(repo/record['path'])==record['sha256']
                    for record in manifest['native_sources']:assert cached_sha(repo.parent/'yquake2'/record['path'])==record['sha256']
                else:
                    assert manifest['sources']==reference['sources'] and manifest['native_sources']==reference['native_sources']
                expected=entry['deterministic_weights_sha256']
                assert (manifest.get('model_weights_sha256') or '').lower()==(expected or '')
                assert manifest['reward_config_sha256'].lower()==task['reward_sha256']
                for name,field in [('q2combat-export.exe','exporter_sha256'),('q2coopbot.exe','client_sha256')]:
                    binary=folder/name;digest=cached_sha(binary);assert digest==manifest[field].lower()
                    binary_examples.setdefault((name,digest),str(binary))
                for record in result['runtime_files']:
                    assert cached_sha(pathlib.Path(result['root'])/'runtime'/record['path'])==record['sha256'].lower()
                members[str(folder)]={'report_sha256':sha(folder/'report.json'),'manifest_sha256':sha(folder/'manifest.json')}
    assert len(source_ids)==len(native_ids)==1
    buildinfos={};normalized={}
    for (name,digest),path in binary_examples.items():
        text=subprocess.check_output(['go','version','-m',path],text=True)
        lines=text.splitlines()[1:]
        common=[line.replace('+dirty','') for line in lines if 'vcs.modified=' not in line]
        if name in normalized:assert common==normalized[name],'Binary build configuration differs beyond dirty stamp'
        else:normalized[name]=common
        buildinfos[digest]={'binary':name,'example':path,'build_info':text}
    recovery=root/'recovery';recovery.mkdir(exist_ok=True)
    proof={'version':'verified_evaluation_members_v1','state':'complete','source_unchanged':True,'jobs':jobs,'members':members,
           'protocol_sha256':sha(root/'protocol.json'),'source_fingerprint':next(iter(source_ids)),
           'native_source_fingerprint':next(iter(native_ids)),'binary_build_infos':buildinfos,
           'scope':'1440 individual capture receipts validated; original aggregate failure retained. All actual binary hashes verified. Distinct build infos differ only in VCS dirty metadata.'}
    save(recovery/'verified-members.json',proof)
    print(json.dumps({'verified':len(members),'binary_variants':len(buildinfos),'proof':str(recovery/'verified-members.json')}))

if __name__=='__main__':main()
