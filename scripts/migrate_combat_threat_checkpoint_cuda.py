"""Verified V8 -> V9 input/Adam expansion; no persisted optimizer step."""
import argparse
import copy
import json
from pathlib import Path
from migrate_combat_navigation_checkpoint_cuda import transfer, verify
from migrate_combat_navigation_cuda import branch, output, sha, torch, validate
from ppo_recurrent import durable_json, durable_write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a = ap.parse_args()
    assert torch.cuda.is_available() and not a.out.exists()
    torch.set_default_device('cuda')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    parent = json.loads((a.source/'migration.json').read_text())
    assert parent['version']=='combat_navigation_checkpoint_migration_v1'
    assert parent['device']=='cuda' and parent['persisted_restoration_verified']
    assert sha(a.source/'weights.json')==parent['weights_sha256']
    assert sha(a.source/'checkpoint.pt')==parent['checkpoint_sha256']
    old = json.loads((a.source/'weights.json').read_text())
    assert old['feature_version']=='combat_features_v8'
    validate(old)
    new = copy.deepcopy(old)
    for role in ('actor','value'):
        for row in new[role][0]['weight']:
            assert len(row)==881
            row.extend([0.]*240)
    new['feature_version']='combat_features_v9'
    validate(new)
    cp = torch.load(a.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    assert cp['weights_sha256']==parent['weights_sha256']
    result,changes = transfer(cp,old,new)
    differences = {}
    with torch.no_grad():
        x = torch.randn(2,12,881,device='cuda')*.1
        x[...,426:466:5]=1
        nx = torch.cat((x,torch.randn(2,12,240,device='cuda')),dim=-1)
        for role in ('actor','value'):
            before,after = output(branch(old,role),x),output(branch(new,role),nx)
            torch.testing.assert_close(before,after,rtol=1e-5,atol=1e-5)
            differences[role]=float((before-after).abs().max())
    checks=verify(result,new)
    a.out.mkdir()
    durable_json(a.out/'weights.json',new)
    result['weights_sha256']=sha(a.out/'weights.json')
    receipt=dict(version='combat_threat_checkpoint_migration_v1',device='cuda',
        gpu=torch.cuda.get_device_name(),source_weights_sha256=parent['weights_sha256'],
        source_checkpoint_sha256=parent['checkpoint_sha256'],parent_receipt_sha256=sha(a.source/'migration.json'),
        weights_sha256=result['weights_sha256'],expanded_parameters=changes,optimizer_checks=checks,
        output_max_error=differences,updates_completed=cp['updates_completed'],
        total_actor_steps=cp['total_actor_steps'],training_performed=False,persisted_optimizer_steps=0)
    result['model_migrations']=cp.get('model_migrations',[])+[copy.deepcopy(receipt)]
    durable_write(a.out/'checkpoint.pt',lambda f:torch.save(result,f))
    restored=torch.load(a.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    verify(restored,new)
    for key in ('config','consumed_rollouts','updates_completed','total_actor_steps','anchor_sha256','bank_sha256'):
        assert restored[key]==cp[key]
    assert torch.equal(restored['rng'],cp['rng'])
    assert len(restored['cuda_rng'])==len(cp['cuda_rng'])
    assert all(torch.equal(x,y) for x,y in zip(restored['cuda_rng'],cp['cuda_rng']))
    assert sha(a.source/'weights.json')==parent['weights_sha256'] and sha(a.source/'checkpoint.pt')==parent['checkpoint_sha256']
    receipt.update(checkpoint_sha256=sha(a.out/'checkpoint.pt'),persisted_restoration_verified=True,
        scope='Append 240 zero columns; keep parent actor/critic, moments, RNG and consumed history. Past observed threats only. No new training or quality claim.')
    durable_json(a.out/'migration.json',receipt)
    durable_json(a.out/'report.json',receipt)
    durable_json(a.out/'complete.json',dict(version='combat_model_migration_complete_v1',
        weights_sha256=receipt['weights_sha256'],checkpoint_sha256=receipt['checkpoint_sha256'],report_sha256=sha(a.out/'report.json')))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
