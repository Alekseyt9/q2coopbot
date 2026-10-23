"""Enable a supported rules control without changing physical fixtures or seeds."""
import argparse
import copy
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('registry','reference','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p):
        p=Path(p).resolve();pins[str(p)]=sha(p);return p
    index=read(pin(a.registry));ref=read(pin(a.reference))
    reference={r['id']:r for r in (read(pin((a.reference.parent/f).resolve())) for f in ref['files'])}
    output=[]
    for f in index['files']:
        original=read(pin((a.registry.parent/f).resolve()));r=reference[original['id']]
        assert 'rules' in r['modes'] and 'learned' in original['modes']
        x,y=copy.deepcopy(original),copy.deepcopy(r)
        for v in (x,y):
            for k in ('revision','modes','splits'):v.pop(k,None)
            v['recipe'].pop('reward_config',None)
        assert x==y,'Reference uses a different physical fixture: '+original['id']
        updated=copy.deepcopy(original);updated['modes']=list(dict.fromkeys([*original['modes'],'rules']));updated['revision']+=1
        check=copy.deepcopy(updated);check['modes']=original['modes'];check['revision']=original['revision'];assert check==original
        output.append((f,updated))
    for p,h in pins.items():assert sha(p)==h,p
    a.out.mkdir(parents=True)
    for f,r in output:assert Path(f).name==f;save(a.out/f,r)
    save(a.out/'index.json',dict(version=index['version'],files=[f for f,_ in output]))
    receipt=a.out.with_name(a.out.name+'-complete.json');assert not receipt.exists()
    save(receipt,dict(state='complete',version='combat_rules_comparison_registry_v1',registry_sha256=sha(a.out/'index.json'),inputs=pins,
         changed_fields=['modes','revision'],physical_fixtures_and_seed_ranges_unchanged=True,training_performed=False,script_sha256=sha(__file__)))
    print(a.out/'index.json')


if __name__=='__main__':main()
