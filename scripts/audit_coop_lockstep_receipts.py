"""Native pair receipts matched against both clients' actual sent commands.

Protocol/serialization audit only; no neural inference and no PPO acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    report=json.loads((args.root/'report.json').read_text(encoding='utf-8-sig'))
    if not report.get('accepted'):raise ValueError('Closed native pair smoke required')
    commands={}
    ends=[]
    for line in (args.root/'server.log').read_text(encoding='utf-8-sig').splitlines():
        if line.startswith('sv_test_pair_cmd '):
            fields={k:int(v) for k,v in (word.split('=',1) for word in line.split()[1:])}
            key=tuple(fields[k] for k in ('spawncount','frame','role','seq','actor'))
            if key in commands:raise ValueError('Duplicate native applied command')
            commands[key]=fields
        if line.startswith('sv_test_pair_step version=1 phase=end '):
            fields={k:int(v) for k,v in (word.split('=',1) for word in line.split()[3:])}
            ends.append(fields)
    traces={}
    paths=[]
    for role,name in enumerate(('PairLearner','PairLeader')):
        path=args.root/(name+'.jsonl');paths.append(path)
        with path.open(encoding='utf-8-sig') as stream:
            for line in stream:
                row=json.loads(line)
                if 'sent_command' not in row:continue
                key=(row['spawncount'],row['observation_frame'],role,row['client_sequence'],row['self_entity'])
                if key in traces:raise ValueError('Duplicate trace command identity')
                traces[key]=row['sent_command']
    if len(ends)!=len(commands) or len(ends)<40 or len(ends)%2:
        raise ValueError('Partial or insufficient native command closure')
    for i,end in enumerate(ends):
        key=(end['spawncount'],end['frame']-1,end['role'],end['seq'],end['actor'])
        if key not in commands or key not in traces:raise ValueError('Unmatched native/trace command')
        actual, sent=commands[key],traces[key]
        for field, tracefield in (('pitch','Pitch'),('yaw','Yaw'),('roll','Roll'),('forward','Forward'),
                                 ('side','Side'),('up','Up'),('buttons','Buttons'),('impulse','Impulse'),
                                 ('msec','Msec'),('light','Light')):
            if actual[field]!=sent[tracefield]:raise ValueError('Native/trace command differs: '+field)
        if end['role']!=i%2:raise ValueError('Native application order differs')
        if i%2 and (end['frame']!=ends[i-1]['frame'] or end['spawncount']!=ends[i-1]['spawncount']):
            raise ValueError('Pair does not share next world frame')
        if i>=2 and end['frame']!=ends[i-2]['frame']+1:
            raise ValueError('World frame gap between paired commands')
    output=dict(version='coop_lockstep_command_audit_v1',completed_pairs=len(ends)//2,
                commands_matched=len(commands),all_command_fields_exact=True,
                source_sha256={str(p):digest(p) for p in paths+[args.root/'server.log',args.root/'report.json']},
                scope='Actual two-client command receipts and consecutive next frames. No reward, reset equivalence, training eligibility or model quality claim.')
    (args.root/'command-receipt-audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(dict(state='passed',pairs=len(ends)//2,commands=len(commands))))


if __name__=='__main__':main()
