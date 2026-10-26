"""CUDA geometry/receipt contract checks on two frozen TRAIN observations."""
import argparse
import copy
import json
from pathlib import Path
from combat_anchor_probe import VERSION, verify
from combat_joint_aim_policy import base_model
from process_combat_architecture_pool import read, save, sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--contexts', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--native-root', type=Path, required=True)
    a=ap.parse_args(); assert not a.out.exists()
    c=read(a.contexts); model=read(c['source_model'])
    assert sha(c['source_model'])==c['source_model_sha256']
    parent=a.native_root.resolve()
    wanted={(v['seed'],v['branch_frame']):v for v in c['contexts']}
    rows={};pins={str(a.contexts.resolve()):sha(a.contexts),c['source_model']:sha(c['source_model'])}
    for path in sorted(parent.glob('export-*-native/native-rollout.jsonl')):
        for line in path.open(encoding='utf-8-sig'):
            row=json.loads(line); key=(row['seed'],row['frame'])
            if key in wanted:
                assert key not in rows; rows[key]=row; pins[str(path)]=sha(path)
    assert rows.keys()==wanted.keys()
    audits=[];rejections=0
    for key,row in rows.items():
        ctx=wanted[key]
        for mode in ('identity','observed_mg'):
            m=copy.deepcopy(model);m['diagnostic_anchor_probe']=dict(version=VERSION,frame=key[1],mode=mode)
            try:base_model(m,allow_shared=True)
            except AssertionError:rejections+=1
            else:raise AssertionError('Diagnostic model admitted to ordinary scoring')
            assert base_model(m,allow_probe=True,allow_shared=True)==base_model(model,allow_shared=True)
            v=copy.deepcopy(row)
            v['sample']['anchor_probe']=dict(version=VERSION,frame=key[1],mode=mode,
                target_entity=1,source_degrees=ctx['record']['selected_request'],
                returned_degrees=ctx['record']['selected_request'] if mode=='identity' else ctx['record']['observed_anchor_request'])
            audits.append(verify(m,[v]))
            for kind in ('missing','tamper','wrong_frame','outside','wrong_weapon','no_recoil'):
                bad=copy.deepcopy(v)
                if kind=='missing':bad['sample'].pop('anchor_probe')
                elif kind=='tamper':bad['sample']['anchor_probe']['returned_degrees'][0]+=.001
                elif kind=='wrong_frame':bad['sample']['anchor_probe']['frame']+=1
                elif kind=='outside':bad['frame']+=1
                elif kind=='wrong_weapon':bad['features'][1125]=0
                else:bad['features'][810]=0
                try:verify(m,[bad])
                except AssertionError:rejections+=1
                else:raise AssertionError('Invalid receipt accepted: '+kind)
    for p,h in pins.items():assert sha(p)==h
    save(a.out,dict(state='passed',device='cuda',contexts=len(rows),modes=audits,rejections=rejections,pins=pins,
        neural_forward_performed=False,training_performed=False,
        scope='Frozen TRAIN feature geometry and receipt validation; native action/export and common-prefix acceptance remain separate.'))
    print(a.out)


if __name__=='__main__':main()
