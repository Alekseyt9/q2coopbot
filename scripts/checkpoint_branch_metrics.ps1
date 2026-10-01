function Get-CheckpointBranchMetrics {
    param([string]$Trace,[int]$Generation,[int]$FirstFrame,[int]$LastFrame)
    $rows=@(Get-Content -LiteralPath $Trace|ForEach-Object {$_|ConvertFrom-Json}|Where-Object {$_.spawncount -eq $Generation -and $_.frame -ge $FirstFrame -and $_.frame -le $LastFrame}|Group-Object frame|ForEach-Object {$_.Group[-1]}|Sort-Object frame)
    if(!$rows.Count){throw 'No branch trace observations in requested interval'}
    $healthLoss=0;$deaths=0;$attackFrames=0;$distance=0.0;$ammoDecreases=@{};$previous=$null;$previousInventory=$null
    foreach($row in $rows){
        if($row.sent_command.Buttons -band 1){$attackFrames++}
        if($previous){
            if($null -ne $row.health -and $null -ne $previous.health){$healthLoss+=[math]::Max(0,[int]$previous.health-[int]$row.health);if($previous.health -gt 0 -and $row.health -le 0){$deaths++}}
            if($row.self.Count -eq 3 -and $previous.self.Count -eq 3){$delta=0.0;for($axis=0;$axis -lt 3;$axis++){$delta+=[math]::Pow($row.self[$axis]-$previous.self[$axis],2)};$distance+=[math]::Sqrt($delta)}
        }
        if($row.inventory_known -and $row.inventory_age_frames -le 1){
            $inventory=@{};foreach($item in $row.inventory){if($item.name -in @('Shells','Bullets','Rockets','Cells','Slugs','Grenades')){$inventory[$item.name]=[int]$item.count}}
            if($previousInventory){foreach($name in $previousInventory.Keys){$count=0;if($inventory.ContainsKey($name)){$count=$inventory[$name]};$drop=[math]::Max(0,$previousInventory[$name]-$count);if($drop){$ammoDecreases[$name]=[int]$ammoDecreases[$name]+$drop}}}
            $previousInventory=$inventory
        }else{$previousInventory=$null}
        $previous=$row
    }
    $health=@($rows|Where-Object {$null -ne $_.health}|ForEach-Object {[int]$_.health})
    return [ordered]@{requested_first_frame=$FirstFrame;requested_last_frame=$LastFrame;observed_first_frame=$rows[0].frame;observed_last_frame=$rows[-1].frame;unique_observed_frames=$rows.Count;missing_frames=($LastFrame-$FirstFrame+1-$rows.Count);simulated_seconds=(($LastFrame-$FirstFrame)/10.0);min_health=$(if($health.Count){($health|Measure-Object -Minimum).Minimum}else{$null});observed_health_loss=$healthLoss;observed_deaths=$deaths;attack_command_frames=$attackFrames;observed_ammo_decreases=$ammoDecreases;travel_distance=$distance}
}

function Get-CheckpointPackageRecords {
    param([string]$Path)
    $root=(Resolve-Path -LiteralPath $Path).Path
    $records=@(foreach($entry in Get-ChildItem -LiteralPath $root -Recurse){
        if($entry.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Checkpoint package cannot contain reparse points'}
        if(!$entry.PSIsContainer){[pscustomobject]@{path=[IO.Path]::GetRelativePath($root,$entry.FullName).Replace('\','/');size=$entry.Length;sha256=(Get-FileHash -LiteralPath $entry.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}}
    })
    return @($records|Sort-Object path)
}
