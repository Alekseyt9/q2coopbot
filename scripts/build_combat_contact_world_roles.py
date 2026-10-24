"""Seal whole-world auxiliary roles from predeclared slots; no outcome selection."""
import argparse
import collections
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('corpus','protocol','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or h==expected,str(path);pins[str(path)]=h;return path
    pin(__file__);p=read(pin(a.protocol));assert p['version']=='combat_memory_contact_refresh_corpus_v1'
    assert p['world_roles']==dict(train_slots=[0,1,2],development_slots=[3])
    technical=Path(p['prior_technical_proof']);tc=read(pin(technical));tr=read(pin(technical.parent/'report.json',tc['report_sha256']));assert tr['gate_passed']
    root=a.corpus.resolve();closed=read(pin(root/'complete.json'));assert closed['native_cases']==closed['cuda_cases']==32
    collection=read(pin(root/'protocol.json',closed['protocol_sha256']));assert collection['train_only'] and not collection['diagnostic_intervention'] and len(collection['assignments'])==1
    assignment=collection['assignments'][0];assert assignment['count']==4 and assignment['split']=='train'
    plan=read(pin(assignment['plan'],assignment['plan_sha256']));assert plan['model_sha256']==p['source_weights_sha256']
    pool=read(pin(root/'pool/report.json',closed['pool_sha256']));assert pool['source_unchanged'] and pool['usable_captures']==32
    worlds=[]
    for job in pool['jobs']:
        task=plan['tasks'][job['task_index']];slot=task['seeds'].index(job['seed']);assert task['split']=='train'
        capture=read(pin(Path(job['root'])/'report.json'));assert capture['capture_complete'] and capture['provenance_valid']
        manifest=read(pin(Path(job['root'])/'manifest.json'));assert manifest['inventory_transport_version']==p['inventory_transport_version'] and manifest['inventory_drop_once_frame']==0
        split='train' if slot in p['world_roles']['train_slots'] else 'development'
        worlds.append(dict(seed=job['seed'],map=task['episode']['map'],episode=task['episode']['id'],slot=slot,split=split,folder=str(root/'processing-train'/f"case-{job['job']:04d}")))
    assert len(worlds)==len({w['seed'] for w in worlds})==32 and collections.Counter(w['split'] for w in worlds)==dict(train=24,development=8)
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True);save(a.out/'report.json',dict(state='complete',version='combat_auxiliary_world_roles_v1',worlds=worlds,protocol_sha256=sha(a.protocol),source_corpus=str(root),source_corpus_complete_sha256=sha(root/'complete.json'),inputs=pins,
        neural_execution=False,training_performed=False,ordinary_training_eligible=False,promotion=False,scope='Roles from predeclared within-family slots before outcomes. No H1-forecast or action labels, no quality-based world filtering.'))
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')));print(dict(train=24,development=8))

if __name__=='__main__':main()
