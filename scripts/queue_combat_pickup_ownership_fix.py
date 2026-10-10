"""Prepare exact reviewable patch now; apply only after sealed source-bound eval."""
import argparse,difflib,os,pathlib,sys
from process_combat_architecture_pool import read,save,sha,run
from run_combat_spatial_ppo import wait_process


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--evaluation',type=pathlib.Path,required=True)
    ap.add_argument('--reference',type=pathlib.Path,required=True); ap.add_argument('--out',type=pathlib.Path,required=True)
    ap.add_argument('--wait-pid',type=int,required=True); a=ap.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[1]; out=a.out.resolve(); evaluation=a.evaluation.resolve()
    assert not out.exists(); out.mkdir()
    target=repo/'internal/bot/combat_policy.go'; tests=repo/'internal/bot/multiweapon_fixture_test.go'
    originals={p:p.read_text(encoding='utf-8') for p in (target,tests)}
    before="if o.Weapon != \"Blaster\" && !(b.campaignEvaluation && (machinegunWeapon(o.Weapon) || shotgunWeapon(o.Weapon))) && !(c.testSynchronous && ((c.testWeaponSwitchFixture == \"parasite_machinegun\" && machinegunWeapon(o.Weapon)) || multiWeaponFixture(c.testWeaponSwitchFixture) && (machinegunWeapon(o.Weapon) || shotgunWeapon(o.Weapon)))) {"
    after="supportedFixture := c.testWeaponSwitchFixture == \"parasite_blaster\" || c.testWeaponSwitchFixture == \"parasite_machinegun\" || multiWeaponFixture(c.testWeaponSwitchFixture)\n\tfixtureWeapon := c.testSynchronous && supportedFixture && (machinegunWeapon(o.Weapon) || shotgunWeapon(o.Weapon))\n\tif o.Weapon != \"Blaster\" && !(b.campaignEvaluation && (machinegunWeapon(o.Weapon) || shotgunWeapon(o.Weapon))) && !fixtureWeapon {"
    oldtest='''\tc.testWeaponSwitchFixture = "parasite_weapons"
\t_, _, sel, direct = c.combatCommand(c.combatObservation(time.Now()), time.Now())
\tif !direct || sel.Owner != "provider" {
\t\tt.Fatal("rules stole Shotgun control", sel)
\t}'''
    newtest='''\tfor _, fixture := range []string{"parasite_blaster", "parasite_machinegun", "parasite_weapons", "parasite_weapons-scarce"} {
\t\tc.testWeaponSwitchFixture = fixture
\t\t_, _, sel, direct = c.combatCommand(c.combatObservation(time.Now()), time.Now())
\t\tif !direct || sel.Owner != "provider" || sel.Fallback != "" {
\t\t\tt.Fatal("rules stole supported picked-up Shotgun control", fixture, sel)
\t\t}
\t}'''
    assert originals[target].count(before)==1 and originals[tests].count(oldtest)==1
    replacements={target:originals[target].replace(before,after),tests:originals[tests].replace(oldtest,newtest).replace('TestShotgunDirectOwnershipRequiresMultiFixture','TestShotgunDirectOwnershipRequiresSupportedFixture')}
    patch=out/'pickup-ownership.patch'
    patch.write_text(''.join(''.join(difflib.unified_diff(originals[p].splitlines(True),replacements[p].splitlines(True),
        fromfile='a/'+p.relative_to(repo).as_posix(),tofile='b/'+p.relative_to(repo).as_posix())) for p in originals),encoding='utf-8',newline='\n')
    hashes={str(p):sha(p) for p in originals}
    save(out/'preparation.json',dict(state='prepared',patch_sha256=sha(patch),original_source_sha256=hashes,
        intent='Supported picked-up Shotgun/Machinegun stays under learned control in synchronous Blaster/MG fixtures; unsupported/outside-pilot gates unchanged. Test provider is fixed, no NN inference.'))
    try:
        wait_process(a.wait_pid,out/'progress.json','waiting_for_frozen_evaluation')
        state=read(evaluation/'progress.json')
        assert state['stage']=='complete' and state['diagnostics_complete']
        assert state['quality_report_sha256']==sha(evaluation/'quality-report.json')
        assert state['diagnostics_acceptance_sha256']==sha(evaluation/'diagnostics-acceptance.json')
        for p in originals: assert sha(p)==hashes[str(p)], 'Source edited while patch queued'
        assert sha(patch)==read(out/'preparation.json')['patch_sha256']
        save(out/'progress.json',dict(stage='applying_ownership_fix',evaluation_quality_sha256=sha(evaluation/'quality-report.json')))
        os.chdir(repo)
        run(['git','apply','--check',patch],out/'patch-check.log')
        run(['git','apply',patch],out/'patch-apply.log')
        run(['gofmt','-w',target,tests],out/'gofmt.log')
        os.environ['GOCACHE']=str(repo/'workspace/build/go-cache'); os.environ['GOTOOLCHAIN']='auto'
        os.environ['Q2_SEARCH_SCAN_ROOT']=str(repo.parent/'assets/baseq2')
        save(out/'source-after.json',{str(p):sha(p) for p in originals})
        run(['go','test','./internal/bot','-count=1','-v','-run',
            '^Test(ShotgunDirectOwnershipRequiresSupportedFixture|MachinegunDirectOwnershipRequiresItsIsolatedFixture|ShotgunBarrelSpreadGuardPreservesPolicyAimAndMovement|MachinegunBarrelSpreadGuardOnlyStopsFire)$'],out/'contract-tests.log')
        testlog=(out/'contract-tests.log').read_text(encoding='utf-8')
        assert '--- SKIP' not in testlog and testlog.count('--- PASS: Test')>=4
        save(out/'progress.json',dict(stage='running_live_pickup_validation',contract_tests_sha256=sha(out/'contract-tests.log')))
        live=out/'live-validation'
        run([sys.executable,repo/'scripts/run_combat_sampling_evaluation.py','--reference',a.reference.resolve(),'--out',live],out/'live-validation.log')
        assert read(live/'progress.json')['diagnostics_complete']
        run([sys.executable,repo/'scripts/report_combat_control_ownership.py','--root',live],out/'ownership.log')
        ownership=read(live/'control-ownership.json')
        learned={k:v for k,v in ownership['groups'].items() if k!='rules-baseline'}
        assert all(not v['fallback_reasons'].get('pilot_equip_not_ready',0) for v in learned.values())
        assert all(not v['counts'].get('rules_with_clear_target',0) for v in learned.values())
        save(out/'progress.json',dict(stage='complete',contract_tests_sha256=sha(out/'contract-tests.log'),
            live_quality_sha256=sha(live/'quality-report.json'),control_ownership_sha256=sha(live/'control-ownership.json'),
            rule_attack_with_clear_target=0,promotion=None))
    except Exception as error:
        save(out/'progress.json',dict(stage='failed',error=str(error),promotion=None)); raise


if __name__=='__main__': main()
