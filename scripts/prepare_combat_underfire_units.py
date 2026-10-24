"""Choose observed past-damage roots and seal movement variants on CUDA."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or h==expected,str(path)
        pins[str(path)]=h;return path
    p=read(pin(a.protocol));assert p['version'] in ('combat_movement_underfire_pairs_v1','combat_movement_underfire_seed_units_v1')
    teacher=p['version']=='combat_movement_underfire_seed_units_v1'
    dataset=Path(p['dataset']).resolve();seal=read(pin(dataset/'complete.json'))
    ds=read(pin(dataset/'report.json',seal['report_sha256']))
    assert ds['version']==('combat_underfire_teacher_dataset_v1' if teacher else 'combat_enemy_threat_fresh_dataset_v1') and ds['source_weights_sha256']==p['source_weights_sha256']
    contexts=read(pin(dataset/'contexts.json',ds['contexts_sha256']))['contexts']
    selected=[min((c for c in contexts if c['instance']['map']==mp and c['split']=='train'),key=lambda c:c['seed']) for mp in p['maps']]
    assert len(selected)==len({c['seed'] for c in selected})==4
    source=Path(p['source']).resolve();pin(source/'weights.json',p['source_weights_sha256'])
    corpus=Path(p['corpus']).resolve();closed=read(pin(corpus/'complete.json'))
    assert closed['native_cases']==closed['cuda_cases']==32
    pin(p['registry']);helper=pin(Path(__file__).with_name('prepare_combat_movement_window_cuda.py'))
    a.out.mkdir(parents=True);units=[]
    for c in selected:
        assert c['role']==('auxiliary_train' if teacher else 'fresh_audit_not_fitting') and c['observed_health_drop']>0
        cid=c['context_id'];folder=a.out/cid;folder.mkdir();metadata=folder/'context';metadata.mkdir()
        report=dict(state='complete',version='combat_observed_damage_context_v1',device='metadata_only',context=c,
            source=str(source),baseline=str(corpus),registry=str(Path(p['registry']).resolve()),inputs=copy.deepcopy(pins),
            training_performed=False,neural_execution=False)
        save(metadata/'report.json',report);save(metadata/'complete.json',dict(state='complete',report_sha256=sha(metadata/'report.json')))
        unit_protocol=dict(version='combat_movement_window_smoke_v1',context_id=cid,variants=p['variants'],
            source_weights_sha256=p['source_weights_sha256'],native_cases=len(p['variants']),cuda_cases=len(p['variants']),fresh_ordinary_reference=False,
            parent_protocol=str(a.protocol.resolve()),parent_protocol_sha256=sha(a.protocol),max_instances=16,timescale=2)
        save(folder/'protocol.json',unit_protocol)
        target=folder/'cuda'
        with (folder/'prepare.log').open('w',encoding='utf-8') as log:
            done=subprocess.run([sys.executable,str(helper),'--prepared',str(metadata),'--protocol',str(folder/'protocol.json'),'--out',str(target)],
                stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert done.returncode==0,str(folder/'prepare.log')
        complete=read(pin(target/'complete.json'));prepared=read(pin(target/'report.json',complete['report_sha256']))
        assert prepared['device']=='cuda' and prepared['neural_parameters_exact'] and prepared['zero_control_exact']
        units.append(dict(context=c,prepared=str(target.resolve()),unit_id=cid))
        save(a.out/'progress.json',dict(stage='cuda-preparing',contexts_done=len(units),contexts=4))
    for path,h in pins.items():assert sha(path)==h,path
    report=dict(state='complete',version='combat_movement_units_preparation_v1',device='cuda',units=units,
        protocol_sha256=sha(a.protocol),source_weights_sha256=p['source_weights_sha256'],inputs=pins,
        native_cases=4*len(p['variants']),cuda_cases=4*len(p['variants']),training_performed=False,promotion=False)
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    save(a.out/'progress.json',dict(stage='complete',contexts=4))
    print(dict(contexts=4,variants=4*len(p['variants']),device='cuda'))


if __name__=='__main__':main()
