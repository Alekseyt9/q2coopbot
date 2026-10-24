"""Select geometric ground-command candidates from sealed old TRAIN captures."""
import argparse
import collections
import math
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists()
    seal=read(a.source/'complete.json');assert seal['state']=='complete'
    assert sha(a.source/'report.json')==seal['report_sha256']
    report=read(a.source/'report.json');assert report['mechanical_reconstruction_passed']
    assert sha(a.source/'request.json')==report['request_sha256']
    assert sha(a.source/'geometry.json')==report['geometry_sha256']
    for p,h in report['inputs'].items():assert sha(p)==h,p
    entries={e['id']:e for e in read(a.source/'request.json')['entries']}
    labels=read(a.source/'labels.json');candidates=[]
    for item in read(a.source/'geometry.json')['results']:
        audit=item['audit']
        if not audit['proposed_path_checked']:continue
        e=entries[item['id']];l=labels[item['id']];cmd=e['proposed'];applied=e['applied'];s=e['snapshot']
        assert s['on_ground'] and s['health']>0 and cmd['Up']<0
        assert applied['Forward']==applied['Side']==applied['Up']==0
        assert cmd['Forward']!=0 or cmd['Side']!=0
        assert l['reasons']==['unsupported_motion_guard']
        assert audit['proposed_prediction'] and audit['applied_prediction']
        candidates.append(dict(id=item['id'],map=s['map'],corpus=l['corpus'],seed=l['seed'],frame=l['frame'],episode=l['episode'],
            vertical_velocity=s['self_velocity'][2],native_control_comparable=l['native_comparable'],
            proposed_control_distance=math.dist(audit['proposed_prediction']['position'],audit['applied_prediction']['position']),
            contacts=len(audit['proposed_prediction']['contacts'] or []),requested=cmd,applied=applied,
            candidate_prediction=audit['proposed_prediction'],control_prediction=audit['applied_prediction']))
    assert len(candidates)==63
    # Frozen first open-floor state per city1/jail1 and first contact state on
    # city1 from the old common initializer corpus. No battle outcome selection.
    source=[c for c in candidates if 'combat-curriculum-balance-v1-20261024' in c['id']
            and c['native_control_comparable'] and c['proposed_control_distance']>=1]
    selected=[]
    for name,contact in [('city1',False),('jail1',False),('city1',True)]:
        eligible=[c for c in source if c['map']==name and bool(c['contacts'])==contact]
        assert eligible
        selected.append(min(eligible,key=lambda c:(c['frame'],c['seed'])))
    assert len({c['id'] for c in selected})==3
    a.out.mkdir(parents=True)
    save(a.out/'report.json',dict(state='complete',version='combat_checked_ground_candidates_v1',
        source_report_sha256=sha(a.source/'report.json'),source_complete_sha256=sha(a.source/'complete.json'),
        candidates=candidates,selected=selected,counts_by_map=dict(collections.Counter(c['map'] for c in candidates)),
        neural_execution=False,new_games=False,training_performed=False,promotion=False,
        scope='63 counterfactual commands from 638 prior crouch-cancellation TRAIN records; not a census of all movement rejections. '
              'Same existing detailed Pmove/path checks; no native execution proof for requested commands yet. '
              'Three states selected before new outcomes by frame then seed within explicit map/contact strata.'))
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(candidates=len(candidates),selected=[c['id'] for c in selected]))


if __name__=='__main__':main()
