[CmdletBinding()]
param([Parameter(Mandatory)][string]$RulesRoot,[Parameter(Mandatory)][string]$LearnedRoot,[Parameter(Mandatory)][string]$Out)
$ErrorActionPreference='Stop'
if(Test-Path -LiteralPath $Out){throw 'Fresh comparison report required'}
$roots=@((Resolve-Path -LiteralPath $RulesRoot).Path,(Resolve-Path -LiteralPath $LearnedRoot).Path)
$reports=@($roots|ForEach-Object {Get-Content -LiteralPath (Join-Path $_ 'report.json') -Raw|ConvertFrom-Json})
$manifests=@($roots|ForEach-Object {Get-Content -LiteralPath (Join-Path $_ 'manifest.json') -Raw|ConvertFrom-Json})
foreach($r in $reports){if(!$r.provenance_valid -or !$r.capture_complete){throw 'Capture/provenance must be valid'}}
if($manifests[0].provider -ne 'rules' -or $manifests[1].provider -ne 'learned'){throw 'Expected rules versus direct learned'}
foreach($field in @('source_fingerprint','native_source_fingerprint','timescale','game_frames','workers','episodes_per_worker','loadout','mixed','health_kit','synchronous','teacher_vertical','reward_config_sha256')){
    if($manifests[0].$field -ne $manifests[1].$field){throw "Comparison differs in $field"}
}
$seeds=@($reports[0].results.seed|Sort-Object)
if(($seeds -join ',') -ne (@($reports[1].results.seed|Sort-Object) -join ',')){throw 'Paired seeds required'}
if(@($seeds|Select-Object -Unique).Count -ne $seeds.Count){throw 'Duplicate episode seed'}
function Metrics($r){
    $damage=0.0;$kills=0
    foreach($target in $r.first_life.outgoing_by_target_class){$damage+=$target.health_damage;$kills+=$target.kills}
    [ordered]@{first_life_end=$r.first_life.end_reason;received_health_damage=$r.first_life.damage.received_health_damage;monster_health_damage=$damage;monster_kills=$kills;experimental_reward=$r.dataset.reward_sum;provider_frames=$r.provider_controlled_frames;guard_interventions=$r.guard_interventions;selection_p99_us=$r.selection_p99_us;capture_mismatches=$r.capture_mismatches;gameplay_accepted=$r.harness_accepted}
}
$pairs=@(foreach($seed in $seeds){
    $a=@($reports[0].results|Where-Object seed -eq $seed);$b=@($reports[1].results|Where-Object seed -eq $seed)
    if($a.Count -ne 1 -or $b.Count -ne 1 -or !$a[0].seed_confirmed -or !$b[0].seed_confirmed){throw 'Invalid seed pairing'}
    if(!$a[0].capture_valid -or !$b[0].capture_valid -or !$a[0].dispatch_valid -or !$b[0].dispatch_valid){throw 'Invalid command proof'}
    [ordered]@{seed=$seed;rules=(Metrics $a[0]);learned=(Metrics $b[0])}
})
$result=[ordered]@{version='combat_bc_comparison_v1';scope='Paired cold episodes, first observed life; full-world reset equivalence unproven. Experimental reward is not win rate. Rules retain setup/death/noncombat; learned commands retain guards.';rules_root=$roots[0];learned_root=$roots[1];rules_report_sha256=(Get-FileHash (Join-Path $roots[0] 'report.json')).Hash;learned_report_sha256=(Get-FileHash (Join-Path $roots[1] 'report.json')).Hash;weights_sha256=$manifests[1].probe_sha256;pairs=$pairs}
$result|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $Out -Encoding utf8NoBOM
"Comparison: $Out"
