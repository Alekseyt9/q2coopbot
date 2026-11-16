"""Independent float64 actor/weapon likelihood and raw-latent native audit."""
import argparse
import json
import math
import pathlib
import torch
from torch.nn import functional as F
from train_combat_hjepa import sha,save,serialized_forward
from audit_combat_hjepa_weapons import available,NAMES


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True)
    a=ap.parse_args();mp=pathlib.Path(str(a.data)+'.meta.json');meta=json.loads(mp.read_text())
    if meta['version']!='hjepa_onpolicy_actor_rollout_v1' or meta['split']!='train' or meta['data_sha256']!=sha(a.data):raise ValueError('typed fresh actor export required')
    for name,digest in meta['sources_sha256'].items():
        if sha(pathlib.Path(name))!=digest:raise ValueError('source changed')
    model=json.loads(pathlib.Path(meta['model']).read_text())
    if sha(pathlib.Path(meta['model']))!=meta['model_sha256'] or model.get('motor') or model['planner']!='bc':raise ValueError('actor source contract')
    cfg=model['primitive_sampling']
    if cfg['version']!='hjepa_primitive_distribution_v1':raise ValueError('primitive distribution changed')
    rows=[json.loads(l) for l in a.data.open()]
    if len(rows)!=meta['rows']:raise ValueError('export row count')
    torch.set_num_threads(2);maximum=dict(raw_actor=0.,primitive_log_probability=0.,joint_log_probability=0.,weapon_probabilities=0.,proposed_action=0.)
    seen=set();current_source=None;source_steps={};counts=dict(attack=0,release=0,jump=0,crouch=0,saturated_continuous=0)
    with torch.no_grad():
        for row in rows:
            key=(row['source_sha256'],row['step_index'])
            if key in seen:raise ValueError('duplicate actor transition')
            seen.add(key);s=row['sample']
            if s['version']!='hjepa_primitive_sample_v1' or s['sampling_seed']!=cfg['seed']:raise ValueError('wrong sampled actor receipt')
            x=torch.tensor([row['features']],dtype=torch.float64)
            z=serialized_forward(x,model['levels'][0]['encoder']);raw=serialized_forward(z,model['actor'])[0]
            maximum['raw_actor']=max(maximum['raw_actor'],float((raw-torch.tensor(row['actor_raw'],dtype=torch.float64)).abs().max()))
            latent=torch.tensor(s['latent'],dtype=torch.float64);log_std=torch.tensor(cfg['log_std'],dtype=torch.float64)
            lp=(-.5*((latent-raw[:4])/log_std.exp()).square()-log_std-.5*math.log(2*math.pi)).sum()
            fire=raw[4]/cfg['categorical_temperature'];lp+=F.logsigmoid(fire if s['attack'] else -fire)
            lp+=F.log_softmax(raw[5:]/cfg['categorical_temperature'],0)[s['vertical']]
            maximum['primitive_log_probability']=max(maximum['primitive_log_probability'],abs(float(lp)-row['primitive_log_probability']))
            if row['source']!=current_source:
                current_source=row['source'];source_steps={step['index']:step for step in map(json.loads,pathlib.Path(current_source).open())}
            step=source_steps[row['step_index']];o=step['observation'];action=step['action']
            if o['identity']!=row['identity']:raise ValueError('observation identity changed')
            proposed=[math.tanh(v) for v in s['latent']];actual=[action['forward'],action['side'],action['yaw_delta_degrees']/180,action['pitch_delta_degrees']/180]
            maximum['proposed_action']=max(maximum['proposed_action'],max(abs(v-w) for v,w in zip(proposed,actual)))
            if action['attack']!=s['attack'] or action['vertical']!=['release','jump','crouch'][s['vertical']]:raise ValueError('categorical action mismatch')
            if model.get('weapon_head'):
                wx=x if model.get('weapon_feature_version')=='hjepa_weapon_features_v1' else x[:,:845]
                logits=serialized_forward(wx,model['weapon_actor'])[0]
                mask=torch.tensor([available(o,name) for name in NAMES]);logits=logits.masked_fill(~mask,-torch.inf)
                probabilities=logits.softmax(0);slot=s.get('weapon',0)
                if not mask[slot]:raise ValueError('unavailable sampled weapon')
                if model.get('weapon_sampling_seed') is not None:
                    maximum['weapon_probabilities']=max(maximum['weapon_probabilities'],float((probabilities-torch.tensor(row['weapon_probabilities'],dtype=torch.float64)).abs().max()))
                    lp+=probabilities[slot].log()
                elif slot!=int(logits.argmax()):raise ValueError('greedy weapon mismatch')
            maximum['joint_log_probability']=max(maximum['joint_log_probability'],abs(float(lp)-row['behavior_log_probability']))
            counts['attack']+=int(s['attack']);counts[['release','jump','crouch'][s['vertical']]]+=1
            counts['saturated_continuous']+=sum(abs(v)==1 for v in proposed)
    passed=maximum['raw_actor']<1e-8 and maximum['primitive_log_probability']<1e-7 and maximum['joint_log_probability']<1e-7 and maximum['weapon_probabilities']<1e-8 and maximum['proposed_action']<1e-12
    report=dict(version='hjepa_native_actor_likelihood_audit_v1',passed=passed,rows=len(rows),worlds=len(meta['episodes']),max_abs_error=maximum,counts=counts,
                data_sha256=sha(a.data),meta_sha256=sha(mp),scope='Independent serialized float64 shared encoder/actor and masked weapon probabilities. Gaussian density in stored raw latent space, Bernoulli fire and categorical pose at declared temperature. Proposed commands, not guarded applied actions, define behavior likelihood. Source outcomes are not neural inputs. Connectivity/parity proof, no improvement or PPO-update claim.')
    save(a.out,report);print(json.dumps(report,indent=2))
    if not passed:raise ValueError('actor likelihood parity failed')


if __name__=='__main__':main()
