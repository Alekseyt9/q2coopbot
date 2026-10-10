"""Recover terminal infrastructure failures without replacing valid outcomes.

Original receipts remain immutable. A derived evaluation uses directory junctions
to original successes and accepted retries; physical trace roots stay unchanged.
"""
import argparse, copy, json, os, pathlib, subprocess, sys
from process_combat_architecture_pool import read, save, sha, run, member
from run_combat_target_refresh import pool


def key(job):
    return (job['plan_index'], job['task_index'], job['seed'], job['mode'])


def validate(job, entry, repo, reference=None):
    folder, report, manifest = member(job['root'], job['seed'])
    plan = read(entry['plan']); task = plan['tasks'][job['task_index']]
    row = report['results'][0]
    assert row['frame_budget_valid']
    assert (manifest.get('model_weights_sha256') or '').lower() == (entry['deterministic_weights_sha256'] or '')
    assert manifest['reward_config_sha256'].lower() == task['reward_sha256']
    assert sha(folder/'q2coopbot.exe') == manifest['client_sha256'].lower()
    if task.get('instances'):
        assert row['generated_fixture'] == task['instances'][task['seeds'].index(job['seed'])]
        assert row['generated_start']['confirmed']
    if reference:
        for field in ('sources', 'native_sources', 'source_fingerprint', 'native_source_fingerprint'):
            assert manifest[field] == reference[field], field
    else:
        for field, base in [('sources', repo), ('native_sources', repo.parent/'yquake2')]:
            for record in manifest[field]:
                assert sha(base/record['path']) == record['sha256'], record['path']
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    a = ap.parse_args(); repo = pathlib.Path(__file__).resolve().parents[1]
    root, out = a.root.resolve(), a.out.resolve()
    artifacts = (repo/'workspace/artifacts').resolve()
    assert root.is_relative_to(artifacts) and out.is_relative_to(artifacts)
    assert read(root/'progress.json')['stage'] == 'failed' and not out.exists()
    protocol = read(root/'protocol.json')
    paths = sorted((root/'pool/jobs').glob('*-result.json'))
    jobs = [read(p) for p in paths]
    assert len(jobs) == protocol['total_episodes'] == len({key(j) for j in jobs})
    failed = {key(j) for j in jobs if j['error']}; assert failed
    # This recovery accepts disk-capacity infrastructure errors only.
    assert all('not enough space on the disk' in j['error'].lower() for j in jobs if j['error'])
    for entry in protocol['evaluations']:
        assert sha(entry['plan']) == entry['plan_sha256']
        plan = read(entry['plan'])
        assert sha(plan['registry_path']) == plan['registry_sha256']
        if plan.get('model_path'): assert sha(plan['model_path']) == entry['deterministic_weights_sha256']
    reference = None
    for job in jobs:
        if not job['error']:
            reference = validate(job, protocol['evaluations'][job['plan_index']], repo, reference)
    assert reference is not None
    out.mkdir(); (out/'original-jobs').mkdir()
    for path in paths: (out/'original-jobs'/path.name).write_bytes(path.read_bytes())
    save(out/'original-receipts.json', dict(protocol_sha256=sha(root/'protocol.json'),
        jobs={str(p): sha(p) for p in paths}, failed_keys=[list(k) for k in sorted(failed)],
        scope='Original successful outcomes retained; only terminal disk failures eligible for replacement.'))
    save(out/'progress.json', dict(stage='retrying',retained=len(jobs)-len(failed),replacements=len(failed)))
    os.environ['GOCACHE'] = str(repo/'workspace/build/go-cache')
    os.environ['GOTOOLCHAIN'] = 'auto'
    pwsh = repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe'
    try:
        plans=[]; mapping={}
        for pi,entry in enumerate(protocol['evaluations']):
            plan=read(entry['plan']); retry=copy.deepcopy(plan); retry['tasks']=[]
            retry['output_root']=str(out/f'attempts-{pi}')
            for ti,task in enumerate(plan['tasks']):
                if not any(k[0]==pi and k[1]==ti for k in failed): continue
                ri=len(retry['tasks'])
                for mode in task['modes']:
                    for seed in task['seeds']: mapping[(len(plans),ri,seed,mode)]=(pi,ti,seed,mode)
                retry['tasks'].append(copy.deepcopy(task))
            if retry['tasks']:
                path=out/f'retry-plan-{pi}.json'; save(path,retry); plans.append(path)
        retries=pool(repo,pwsh,plans,out/'retry-pool',out/'retry.log')
        replacements={mapping[key(j)]:j for j in retries['jobs'] if mapping[key(j)] in failed}
        assert set(replacements)==failed
        derived=copy.deepcopy(protocol); receipts=[]; accepted=[]
        (out/'pool/jobs').mkdir(parents=True)
        for pi,entry in enumerate(derived['evaluations']):
            directory=out/(entry['model']+'-'+entry['label']); directory.mkdir()
            plan=read(entry['plan']); plan['output_root']=str(directory/'capture')
            path=directory/'plan.json'; save(path,plan)
            entry.update(plan=str(path),plan_sha256=sha(path),root=plan['output_root'])
        for job in jobs:
            replacement=replacements.get(key(job)); physical=replacement or job
            validate(dict(physical,task_index=job['task_index']),protocol['evaluations'][job['plan_index']],repo,reference)
            entry=derived['evaluations'][job['plan_index']]
            alias=pathlib.Path(entry['root'])/f"case-{job['task_index']}-{job['mode']}"/f"s-{job['seed']}"
            alias.parent.mkdir(parents=True,exist_ok=True)
            target=pathlib.Path(physical['root']).resolve()
            assert target.is_relative_to(artifacts) and alias.is_relative_to(out)
            quote=lambda p: "'"+str(p).replace("'","''")+"'"
            run([pwsh,'-NoProfile','-Command',f"New-Item -ItemType Junction -Path {quote(alias)} -Target {quote(target)} -ErrorAction Stop | Out-Null"],out/'junction.log')
            assert os.path.samefile(alias,target)
            accepted_job=dict(job,root=str(alias),error='')
            accepted.append(accepted_job); save(out/'pool/jobs'/f"{job['job']}-result.json",accepted_job)
            receipts.append(dict(original=job,retry=replacement,accepted=accepted_job,
                report_sha256=sha(target/'report.json'),manifest_sha256=sha(target/'manifest.json')))
        derived['recovery']=dict(original_root=str(root),original_protocol_sha256=sha(root/'protocol.json'),
            retry_pool_sha256=sha(out/'retry-pool/report.json'),retained=len(jobs)-len(failed),replaced=len(failed),
            scope='Derived path namespace only; original 49 successes and seven disk failures immutable. Whole-group retries retained; only predefined failures substituted, irrespective of outcome.')
        save(out/'protocol.json',derived); save(out/'recovery-receipts.json',receipts)
        save(out/'pool/report.json',dict(state='complete',source_unchanged=True,jobs=accepted,
            scheduler='independent_episode_queue',recovery=derived['recovery']))
        run([sys.executable,repo/'scripts/verify_combat_evaluation_members.py','--root',out,
            '--source-fingerprint',reference['source_fingerprint'],'--native-fingerprint',reference['native_source_fingerprint']],out/'verify.log')
        run([sys.executable,repo/'scripts/report_combat_architecture_evaluation.py','--root',out,
            '--member-proof',out/'recovery/verified-members.json'],out/'quality.log')
        run([sys.executable,repo/'scripts/finalize_combat_machinegun_evaluation.py','--root',out],out/'diagnostics.log')
        run([sys.executable,repo/'scripts/report_combat_visible_target_decisions.py','--root',out,
            '--families',*protocol['families'],'--out',out/'visible-target-decisions.json'],out/'visible-target-decisions.log')
        for path in paths: assert sha(path)==read(out/'original-receipts.json')['jobs'][str(path)]
        save(out/'recovery-complete.json',dict(state='complete',retained=len(jobs)-len(failed),replaced=len(failed),
            quality_report_sha256=sha(out/'quality-report.json'),original_receipts_unchanged=True,promotion=None))
        print(json.dumps(dict(state='complete',retained=len(jobs)-len(failed),replaced=len(failed))))
    except Exception as error:
        save(out/'progress.json',dict(stage='failed',error=str(error),promotion=None));raise


if __name__=='__main__': main()
