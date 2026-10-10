"""Shared closure for native comparisons; no training or automatic promotion."""
import pathlib
import sys

from process_combat_architecture_pool import read, save, sha, run


def finalize(repo, root, result):
    manifest = read(pathlib.Path(result['jobs'][0]['root']) / 'manifest.json')
    run([sys.executable, repo / 'scripts/verify_combat_evaluation_members.py', '--root', root,
        '--source-fingerprint', manifest['source_fingerprint'], '--native-fingerprint', manifest['native_source_fingerprint']], root / 'verify.log')
    run([sys.executable, repo / 'scripts/audit_combat_sampling_configs.py', '--root', root], root / 'sampling-config.log')
    assert not read(root / 'sampling-config-audit.json')['repeated_stochastic_execution_configs']
    run([sys.executable, repo / 'scripts/report_combat_architecture_evaluation.py', '--root', root,
        '--member-proof', root / 'recovery/verified-members.json'], root / 'quality.log')
    run([sys.executable, repo / 'scripts/finalize_combat_machinegun_evaluation.py', '--root', root], root / 'diagnostics.log')
    run([sys.executable, repo / 'scripts/report_combat_control_ownership.py', '--root', root], root / 'control-ownership.log')
    ownership = read(root / 'control-ownership.json')
    reference = read(root / 'protocol.json')['comparison_reference']
    counts = {name: dict(visible_target_rules=g['counts'].get('rules_with_clear_target', 0),
        equip_fallback=g['fallback_reasons'].get('pilot_equip_not_ready', 0))
        for name, g in ownership['groups'].items() if name != reference}
    save(root / 'ownership-acceptance.json', dict(state='complete', counts=counts,
        clean=all(not v['visible_target_rules'] and not v['equip_fallback'] for v in counts.values()),
        control_ownership_sha256=sha(root / 'control-ownership.json'), promotion=None))
    run([sys.executable, repo / 'scripts/report_combat_outcome_behavior.py', '--root', root], root / 'outcome-behavior.log')
    run([sys.executable, repo / 'scripts/audit_combat_physical_storage.py', '--root', root, '--out', root / 'storage-final.json'], root / 'storage.log')
    status = read(root / 'progress.json')
    status.update(ownership_acceptance_sha256=sha(root / 'ownership-acceptance.json'),
        outcome_behavior_sha256=sha(root / 'outcome-behavior.json'), physical_storage_sha256=sha(root / 'storage-final.json'))
    save(root / 'progress.json', status)
