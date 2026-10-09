"""Repair only failed terminal members through the existing 16-slot pool."""
import argparse,copy,os,pathlib,sys
from process_combat_architecture_pool import read,save,sha,run,member
from run_combat_target_refresh import pool

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--capture-root',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--processing-out',type=pathlib.Path,required=True);a=ap.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[1];capture=a.capture_root.resolve();out=a.out.resolve();processing=a.processing_out.resolve()
    original=read(capture/'pool/report.json');assert original['state']=='failed' and original['source_unchanged'] and not out.exists() and not processing.exists()
    assert read(capture/'execution.json')['stage']=='failed';models=read(capture/'models.json');failed=[j for j in original['jobs'] if j['error']];assert failed
    out.mkdir();save(out/'models.json',models);save(out/'config.json',read(capture/'config.json'))
    save(out/'execution.json',dict(stage='preparing_retries',failed_members=len(failed),original_pool_sha256=sha(capture/'pool/report.json')))
    os.environ['GOCACHE']=str(repo/'workspace/build/go-cache');os.environ['GOTOOLCHAIN']='auto'
    try:
        plans=[];mapping={};failed_keys={(j['plan_index'],j['task_index'],j['seed'],j['mode']) for j in failed}
        for pi,binding in enumerate(models):
            plan=read(binding['plan']);retry=copy.deepcopy(plan);retry['tasks']=[];retry['output_root']=str(out/f'attempts-{pi}')
            for ti,task in enumerate(plan['tasks']):
                seeds=[j['seed'] for j in failed if j['plan_index']==pi and j['task_index']==ti]
                if not seeds:continue
                # Frozen schedules require four-seed groups. Keep that contract;
                # only the failed original members are replaced after the retry.
                new=copy.deepcopy(task)
                for seed in new['seeds']:mapping[(len(plans),len(retry['tasks']),seed)]=(pi,ti,seed,'learned')
                retry['tasks'].append(new)
            if retry['tasks']:
                path=out/f'retry-plan-{pi}.json';save(path,retry);plans.append(path)
        save(out/'execution.json',dict(stage='retrying_native_members',failed_members=len(failed),episodes=sum(len(t['seeds']) for p in plans for t in read(p)['tasks'])))
        retries=pool(repo,repo/'workspace/tools/dev_tools/powershell-7.5.3/runtime/pwsh.exe',plans,out/'retry-pool',out/'retry.log')
        replacements={mapping[(j['plan_index'],j['task_index'],j['seed'])]:j for j in retries['jobs'] if mapping[(j['plan_index'],j['task_index'],j['seed'])] in failed_keys};assert len(replacements)==len(failed)
        repaired=copy.deepcopy(original);receipts=[];sources=set();native_sources=set()
        for job in repaired['jobs']:
            key=(job['plan_index'],job['task_index'],job['seed'],job['mode']);old=copy.deepcopy(job)
            if key in replacements:
                replacement=replacements[key];job.update(root=replacement['root'],error='',recovery_attempt=dict(retry_job=replacement['job'],original_error=old['error']))
                receipts.append(dict(original=old,retry=replacement))
            path,report,manifest=member(job['root'],job['seed']);binding=models[job['plan_index']];plan=read(binding['plan'])
            assert manifest['model_weights_sha256'].lower()==plan['model_sha256']==sha(binding['model'])
            result=report['results'][0];assert result['frame_budget_valid']
            task=plan['tasks'][job['task_index']]
            if task.get('instances'):assert result['generated_fixture']==task['instances'][task['seeds'].index(job['seed'])] and result['generated_start']['confirmed']
            sources.add(manifest['source_fingerprint']);native_sources.add(manifest['native_source_fingerprint'])
        assert len(sources)==len(native_sources)==1 and not any(j['error'] for j in repaired['jobs'])
        repaired.update(state='complete',recovery=dict(original_pool_sha256=sha(capture/'pool/report.json'),retry_pool_sha256=sha(out/'retry-pool/report.json'),replaced_members=len(receipts),scope='Only failed members replaced; original receipts/reports retained, same seed/fixture/model/source contract verified.'))
        (out/'pool').mkdir();save(out/'pool/report.json',repaired);save(out/'recovery-receipts.json',receipts)
        exporter=repo/'workspace/build/q2ppo-data-spatial-v1.exe';run(['go','build','-o',exporter,'./cmd/q2ppo-data'],out/'build-exporter.log')
        save(out/'execution.json',dict(stage='cuda_ppo_processing',replaced_members=len(receipts),accepted_members=len(repaired['jobs'])))
        legacy=repo/'workspace/artifacts/aproc-v1-20261007'
        run([sys.executable,repo/'scripts/process_combat_architecture_pool.py','--capture-root',out,'--out',processing,'--config',out/'config.json','--exporter',exporter,'--anchor',legacy/'anchor.json','--bank',legacy/'bank.json','--export-workers',4],out/'process.log')
        result=read(processing/'report.json');assert result['state']=='complete' and len(result['training'])==2 and all(r['device']=='cuda' for r in result['training'])
        save(out/'execution.json',dict(stage='complete',processing_report_sha256=sha(processing/'report.json'),promotion='Pending paired validation'))
    except Exception as error:
        save(out/'execution.json',dict(stage='failed',error=str(error),promotion='None; original and retry artifacts retained'));raise

if __name__=='__main__':main()
