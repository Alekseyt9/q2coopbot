[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$TracePath,[int]$StartFrame=0,[int]$EndFrame=0)
$ErrorActionPreference='Stop'
$rows=@(Get-Content -LiteralPath $TracePath|ConvertFrom-Json|Where-Object {$_.self -and $_.frame -ge $StartFrame -and (!$EndFrame -or $_.frame -le $EndFrame)})
if($rows.Count -lt 2){throw 'Return trace has fewer than two observations'}
$path=0.0;$moving=0;$stationary=0;$stall=0;$longest=0;$stalls=0;$stallDetails=@()
for($i=1;$i -lt $rows.Count;$i++){
    $dx=[double]$rows[$i].self[0]-[double]$rows[$i-1].self[0]
    $dy=[double]$rows[$i].self[1]-[double]$rows[$i-1].self[1]
    $step=[math]::Sqrt($dx*$dx+$dy*$dy)
    $path+=$step
    if($step -ge 1){
        $moving++
        if($stall -ge 5){
            $stalls++
            $startIndex=$i-1-$stall;$endIndex=$i-1
            $first=$rows[$startIndex];$last=$rows[$endIndex]
            $reason=@($rows[$startIndex..$endIndex]|ForEach-Object {$_.arbitration.limit_reason}|Where-Object {$_}|Group-Object|Sort-Object Count -Descending|Select-Object -First 1)
            $stallDetails+=@{start_frame=$first.frame;end_frame=$last.frame;frames=$stall;position=$first.self;goal=$first.goal;reason=$(if($reason.Count){$reason[0].Name}else{''})}
        }
        if($stall -gt $longest){$longest=$stall}
        $stall=0
    }else{
        $stationary++
        $stall++
    }
}
if($stall -ge 5){
    $stalls++
    $startIndex=$rows.Count-1-$stall;$endIndex=$rows.Count-1
    $first=$rows[$startIndex];$last=$rows[$endIndex]
    $reason=@($rows[$startIndex..$endIndex]|ForEach-Object {$_.arbitration.limit_reason}|Where-Object {$_}|Group-Object|Sort-Object Count -Descending|Select-Object -First 1)
    $stallDetails+=@{start_frame=$first.frame;end_frame=$last.frame;frames=$stall;position=$first.self;goal=$first.goal;reason=$(if($reason.Count){$reason[0].Name}else{''})}
}
if($stall -gt $longest){$longest=$stall}
$netX=[double]$rows[-1].self[0]-[double]$rows[0].self[0]
$netY=[double]$rows[-1].self[1]-[double]$rows[0].self[1]
$net=[math]::Sqrt($netX*$netX+$netY*$netY)
$reasons=@($rows|ForEach-Object {$_.arbitration.limit_reason})
$sources=@($rows|ForEach-Object {$_.arbitration.move_source})
$jumpFailures=@($rows|Where-Object {$_.arbitration.limit_reason -in @('runup_lost_ground','jump_missed','jump_aborted')}|Select-Object -First 12|ForEach-Object {
    [pscustomobject]@{frame=$_.frame;reason=$_.arbitration.limit_reason;position=$_.self;velocity=$_.self_velocity;from=$_.jump_plan.from;runup=$_.jump_plan.runup;landing=$_.jump_plan.landing;speed=$_.jump_plan.speed;phase=$_.jump_plan.phase}
})
[pscustomobject]@{
    start_frame=$rows[0].frame
    end_frame=$rows[-1].frame
    elapsed_frames=$rows[-1].frame-$rows[0].frame
    horizontal_path=[math]::Round($path,2)
    net_displacement=[math]::Round($net,2)
    path_over_net=if($net -gt 0){[math]::Round($path/$net,3)}else{$null}
    moving_frames=$moving
    stationary_frames=$stationary
    stall_windows_5plus=$stalls
    stall_windows=@($stallDetails|Sort-Object frames -Descending|Select-Object -First 5)
    longest_stall_frames=$longest
    jump_missed=@($reasons|Where-Object {$_ -eq 'jump_missed'}).Count
    runup_lost_ground=@($reasons|Where-Object {$_ -eq 'runup_lost_ground'}).Count
    jump_failures=$jumpFailures
    static_hull_blocked=@($reasons|Where-Object {$_ -eq 'static_hull_blocked'}).Count
    no_verified_landing=@($reasons|Where-Object {$_ -eq 'no_verified_landing'}).Count
    corner_bypass_frames=@($sources|Where-Object {$_ -eq 'route_corner_bypass'}).Count
    elevator_completed=[bool]@($rows|Where-Object {$_.elevator -eq 'completed'}).Count
}
