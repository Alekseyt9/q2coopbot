[CmdletBinding()]
param([Parameter(Mandatory)][string]$Report)
$ErrorActionPreference='Stop'
$reportPath=(Resolve-Path $Report).Path
$out=Split-Path $reportPath -Parent
$items=@(Get-Content $reportPath -Raw|ConvertFrom-Json)
if(!$items.Count -or @($items|Where-Object {!$_.accepted}).Count){throw 'Accepted native report required'}
$file=Get-ChildItem $items[0].artifacts -Filter '*-trace.jsonl'|Where-Object Name -NotLike '*human*'|Select-Object -First 1
$original=@(Get-Content $file.FullName|ConvertFrom-Json)
$repo=Split-Path $PSScriptRoot -Parent
$runtime=Join-Path $repo 'workspace/runtime/q2go-grenade-moving-contact/baseq2'
$exe=Join-Path $out 'q2grenade-report.exe'
$results=@()
foreach($case in @('missing-explosion','wrong-explosion-position','late-explosion','static-target','changed-identity','missing-heldout','changed-gravity')) {
    $rows=@($original|ConvertTo-Json -Depth 100|ConvertFrom-Json)
    $flight=@($rows|Where-Object {$_.projectiles.Count})
    switch($case) {
        'missing-explosion' {foreach($row in $rows){$row.PSObject.Properties.Remove('explosions')}}
        'wrong-explosion-position' {foreach($row in $rows){foreach($event in $row.explosions){$event.position[0]+=100}}}
        'late-explosion' {
            $event=@($rows.explosions)|Select-Object -First 1
            foreach($row in $rows){$row.PSObject.Properties.Remove('explosions')}
            $row=$rows|Where-Object frame -EQ 184|Select-Object -First 1
            $row|Add-Member -NotePropertyName explosions -NotePropertyValue @($event) -Force
        }
        'static-target' {foreach($row in $rows){foreach($target in $row.enemies){$target.origin[1]=-80}}}
        'changed-identity' {$flight[2].enemies[0].id+=10000}
        'missing-heldout' {$flight[5].PSObject.Properties.Remove('projectiles')}
        'changed-gravity' {$flight[2].gravity=600}
    }
    $inputFile=Join-Path $out ('negative-'+$case+'.jsonl')
    $outputFile=Join-Path $out ('negative-'+$case+'.json')
    $rows|ForEach-Object {ConvertTo-Json -InputObject $_ -Depth 100 -Compress}|Set-Content $inputFile -Encoding utf8
    & $exe -trace $inputFile -root $runtime -moving-contact -out $outputFile 2>&1|Out-Null
    $exitCode=$LASTEXITCODE
    $validation=Get-Content $outputFile -Raw|ConvertFrom-Json
    $results+=[pscustomobject]@{case=$case;rejected=($exitCode -ne 0 -and !$validation.accepted);reason=$validation.reason}
}
$results|ConvertTo-Json -Depth 6|Set-Content (Join-Path $out 'negative-controls.json') -Encoding utf8
if(@($results|Where-Object {!$_.rejected}).Count){throw 'Damaged moving contact trace accepted'}
Write-Host "Moving contact: $($results.Count)/$($results.Count) negative controls rejected"
