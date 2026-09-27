[CmdletBinding()]
param([Parameter(Mandatory)][string]$Summary)
$ErrorActionPreference='Stop'
$report=@(foreach($run in @(Get-Content $Summary -Raw|ConvertFrom-Json)){
    $rows=@(Get-Content $run.trace_jsonl|ConvertFrom-Json)
    $before=@($rows|Where-Object map -eq base1);$after=@($rows|Where-Object map -eq base2)
    if($before.Count -lt 10 -or $after.Count -lt 20 -or @($rows|Where-Object {$_.map -notin @('base1','base2') -or $_.light_source -ne 'bsp_static'}).Count){throw 'Incomplete BSP-lit map transition'}
    if(@($before|Where-Object {$_.sent_command.Light -ne 18 -and $_.sent_command.Light -gt 5}).Count -lt 10){throw 'Initial map lighting was not distinguished'}
    foreach($r in $after){
        if([math]::Abs($r.self[0]-848) -gt 1 -or [math]::Abs($r.self[1]-2292) -gt 1 -or $r.sent_command.Light -ne 18){throw 'Wrong base2 entry position/light'}
    }
    if($before[-1].spawncount -eq $after[0].spawncount){throw 'Server generation did not change'}
    $applied=@{}
    foreach($a in @(Get-Content $run.applied_jsonl|ConvertFrom-Json|Where-Object kind -eq new)){
        $key="$($a.spawncount)/$($a.client_sequence)"
        if($applied.ContainsKey($key)){throw 'Duplicate applied command'}
        $applied[$key]=$a
    }
    foreach($r in $rows){
        $a=$applied["$($r.spawncount)/$($r.client_sequence)"]
        if(!$a -or $a.command.Light -ne $r.sent_command.Light){throw 'Server lighting differs from sent command'}
    }
    [pscustomobject]@{timescale=$run.timescale;accepted=$true;base1_frames=$before.Count;base2_frames=$after.Count;base2_light=18;matched_light_commands=$rows.Count;trace=$run.trace_jsonl;scope='base1_to_base2_entry_not_all_campaign_maps'}
})
$report|ConvertTo-Json -Depth 4|Set-Content (Join-Path (Split-Path $Summary -Parent) 'light-transition-report.json')
$report
