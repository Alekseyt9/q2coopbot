param([Parameter(Mandatory)][string]$OutputRoot,[int]$CaptureSeed,[int]$Port,[switch]$RetryFailed,[switch]$RocketCohort)
$ErrorActionPreference='Stop'
$repo=(Resolve-Path "$PSScriptRoot/../..").Path
$out=(Resolve-Path $OutputRoot).Path
$env:Q2COOPBOT_DIAGNOSTIC_ENGINE_ROOT="$repo/workspace/artifacts/fragment-bank-20261116-224759/engine-objective-v1"
if($CaptureSeed){
    $a=Import-Clixml "$repo/workspace/artifacts/fragment-autonomous-train-20261117-052625/source-4.clixml"
    $a.Remove('CheckpointCapturePhase'); $a.Remove('CheckpointCaptureWeapon')
    $name=if($RetryFailed){"source-retry-$CaptureSeed"}else{"source-$CaptureSeed"}
    $a['Seed']=$CaptureSeed; $a['GeneratedFixture']="$out/fixture-$CaptureSeed.json"; $a['OutputRoot']="$out/$name"
    $a['CheckpointCaptureFrame']=110; $a['Port']=$Port
    if($RocketCohort){
        $a['CheckpointCaptureFrame']=101; $a['CheckpointCapturePhase']='equipped_weapon_v1'
        $a['CheckpointCaptureWeapon']='Rocket Launcher'
    }
    & "$repo/scripts/run_solo_tactical_retreat.ps1" @a
    return
}
$trigger="$repo/workspace/artifacts/lewm-rocket-trigger-v1"
$exclude=@(Get-Content "$trigger/validation-seeds.json" -Raw | ConvertFrom-Json)
$plans=if($RocketCohort){Get-ChildItem $out -Filter 'plan-*.json' -File}else{Get-ChildItem $trigger -Filter 'fixtures-validation-*.json' -File}
$fixtures=foreach($file in $plans) {
    foreach($task in (Get-Content $file -Raw | ConvertFrom-Json).tasks) {
        foreach($fixture in $task.instances) {
            if($fixture.split -eq 'validation' -and $fixture.engine_seed -notin $exclude -and
              (-not $RocketCohort -or $fixture.inventory.initial_weapon -eq 'Rocket Launcher')) { $fixture }
        }
    }
}
$selected=@(foreach($group in ($fixtures | Group-Object map | Sort-Object Name)) {
    $group.Group | Sort-Object -Unique -Property engine_seed | Sort-Object {
        [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes([string]$_.engine_seed)))
    } | Select-Object -First 2
})
$seeds=@($selected.engine_seed)
ConvertTo-Json -InputObject $seeds | Set-Content "$out/validation-seeds.json" -Encoding utf8NoBOM
foreach($fixture in $selected) {
    $fixture | ConvertTo-Json -Depth 16 | Set-Content "$out/fixture-$($fixture.engine_seed).json" -Encoding utf8NoBOM
}
$script=$PSCommandPath
$retry=[bool]$RetryFailed
$rocket=[bool]$RocketCohort
0..($seeds.Count-1) | ForEach-Object -ThrottleLimit 3 -Parallel {
    $i=$_; $ids=$using:seeds; $seed=$ids[$i]; $p=$using:out
    $argsList=@('-NoProfile','-File',$using:script,'-OutputRoot',$p,'-CaptureSeed',$seed,'-Port',(50600+$i))
    if($using:rocket){$argsList+='-RocketCohort'}
    $log="$p/capture-$seed.log"
    if($using:retry){
        $report="$p/source-$seed/report.json"
        if((Test-Path $report) -and (Get-Content $report -Raw | ConvertFrom-Json).combat_checkpoint.accepted){return}
        $argsList+='-RetryFailed'
        $log="$p/capture-retry-$seed.log"
    }
    # Each server and its checkpoint tool need their own environment.
    & pwsh @argsList *> $log
}
