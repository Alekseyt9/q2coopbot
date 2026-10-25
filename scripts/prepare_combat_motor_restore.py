"""Single-field motor restoration and geometry-pinned causal diagnostic selection."""
import argparse
import copy
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--control',required=True,type=Path);ap.add_argument('--original',required=True,type=Path)
    ap.add_argument('--selection',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args();assert not a.out.exists()
    control,original=read(a.control),read(a.original)
    assert control['feature_version']=='combat_features_v13_memory_static_routes'
    assert control['motor_residual']['version']==original['motor_residual']['version']=='combat_observed_motor_residual_v1'
    candidate=copy.deepcopy(control);candidate['motor_residual']=copy.deepcopy(original['motor_residual'])
    changed=[k for k in control if candidate[k]!=control[k]]
    assert changed==['motor_residual'] and candidate.keys()==control.keys()
    assert candidate['memory_route_movement']==control['memory_route_movement']
    selection=read(a.selection)
    assert selection['source_weights_sha256']==sha(a.original)
    selection['cases']=[c for c in selection['cases'] if c['episode']=='campaign-ware2-site-04-group-parasite-infantry-weapons']
    assert len(selection['cases'])==1
    selection['selection']='Previously observed lower-floor regression at one fixed site; known48 plus declared49-51 seeds, no quality generalization.'
    selection['selection_kind']='known_regression_diagnostic'
    a.out.mkdir(parents=True);save(a.out/'weights.json',candidate);save(a.out/'selection.json',selection)
    save(a.out/'report.json',dict(version='combat_motor_restore_single_field_v1',state='passed',
        changed_fields=changed,original_motor_sha256_source=sha(a.original),control_sha256=sha(a.control),
        weights_sha256=sha(a.out/'weights.json'),training_performed=False,promotion=False,
        inputs={str(p.resolve()):sha(p) for p in (a.control,a.original,a.selection)},
        scope='Motor-weight intervention only. Conditional aim/fire may change because they depend on sampled movement. Not learned improvement or isolated locomotion.'))
    print(dict(changed_fields=changed,weights_sha256=sha(a.out/'weights.json')),flush=True)


if __name__=='__main__':main()
