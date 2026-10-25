"""Validate task-boundary identity against recorded native control/zero traces; no NN math."""
import argparse
import copy
from pathlib import Path
from evaluate_registered_combat_models import read
from audit_combat_contact_loss_probe import load_steps,equivalent


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--probe',type=Path,required=True);a=ap.parse_args()
    spec=read(a.probe/'spec.json');audit=read(a.probe/'acceptance-v2/report.json');checked=0;tails=0
    for v in spec['variants']:
        if v['label']!='control':continue
        twin=next(x for x in spec['variants'] if x['context_index']==v['context_index'] and x['replicate']==v['replicate'] and x['label']=='zero')
        load=lambda i:load_steps(next((a.probe/f'case-{i:02d}/native').glob('replay-*/steps.jsonl')))
        left,right=load(v['index']),load(twin['index']);end=audit['results'][v['index']]['end_frame']
        assert equivalent(left,right,before=end+1)
        changed=copy.deepcopy(right);next(s for s in changed if s['observation']['identity']['frame']==v['context']['frame'])['sent_command']['Side']+=1
        assert not equivalent(left,changed,before=end+1),'An in-task command change escaped identity check'
        after=next((i for i,s in enumerate(right) if s['observation']['identity']['frame']>end),None)
        if after is not None:
            changed=copy.deepcopy(right);changed[after]['sent_command']['Side']+=1
            assert equivalent(left,changed,before=end+1),'Post-objective shutdown affected task identity'
            tails+=1
        checked+=1
    assert checked==6 and tails>0
    print(dict(passed=True,control_pairs=checked,post_goal_tail_cases=tails,neural_execution=False))


if __name__=='__main__':main()
