[CmdletBinding()]
param([ValidateRange(1024,65530)][int]$BasePort=31830,[string]$OutputRoot='')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
if(!$OutputRoot){$OutputRoot=Join-Path $repo ('workspace/artifacts/return-regression-'+(Get-Date -Format yyyyMMdd-HHmmss-fff))}
$root=[IO.Path]::GetFullPath($(if([IO.Path]::IsPathRooted($OutputRoot)){$OutputRoot}else{Join-Path (Get-Location) $OutputRoot}))
if(Test-Path -LiteralPath $root){throw "Output already exists: $root"}
foreach($port in $BasePort..($BasePort+5)){
    if(Get-NetUDPEndpoint -LocalPort $port -ErrorAction SilentlyContinue){throw "UDP port occupied: $port"}
}
New-Item -ItemType Directory $root|Out-Null
$cases=@(
    @{map='base1';session='base1-far-death-return-session.json'},
    @{map='base2';session='death-point-memory-restart-session.json'},
    @{map='base3';session='base3-full-death-return-session.json'}
)
$jobs=@();$results=@()
try{
    for($i=0;$i -lt $cases.Count;$i++){
        $case=$cases[$i]
        $session=Join-Path $PSScriptRoot ('scenarios/'+$case.session)
        $dir=Join-Path $root $case.map
        $jobs+=Start-ThreadJob -ArgumentList @((Join-Path $PSScriptRoot 'run_memory_restart_suite.ps1'),$case.map,($BasePort+2*$i),$dir,$session) -ScriptBlock {
            param($script,$map,$port,$dir,$session)
            try{
                & $script -Map $map -BasePort $port -OutputRoot $dir -SessionPath $session | Out-Host
                $report=Get-Content -LiteralPath (Join-Path $dir 'report.json') -Raw|ConvertFrom-Json
                if(!$report.accepted -or $report.timescale -ne 2 -or $report.parallelism -ne 2){throw 'Invalid suite acceptance'}
                $fingerprints=@(foreach($run in 0..1){
                    $trial=Join-Path $dir ('run-{0:D3}' -f $run)
                    $files=@('q2coopbot.exe','runtime/q2ded.exe','runtime/baseq2/game.dll',"runtime/baseq2/maps/$map.aas",$session)
                    foreach($path in $files){
                        $file=if([IO.Path]::IsPathRooted($path)){$path}else{Join-Path $trial $path}
                        $hash=Get-FileHash -LiteralPath $file -Algorithm SHA256
                        @{run=$run;file=$path;sha256=$hash.Hash}
                    }
                    foreach($extension in @('bsp','ent')){
                        $asset=Join-Path $trial "runtime/baseq2/maps/$map.$extension"
                        if(Test-Path -LiteralPath $asset){
                            @{run=$run;file="maps/$map.$extension";sha256=(Get-FileHash -LiteralPath $asset -Algorithm SHA256).Hash}
                        }
                    }
                    foreach($asset in Get-ChildItem -LiteralPath (Join-Path $trial 'runtime/baseq2') -File -Filter '*.pak'){
                        @{run=$run;file=$asset.Name;sha256=(Get-FileHash -LiteralPath $asset.FullName -Algorithm SHA256).Hash}
                    }
                })
                @{map=$map;accepted=$true;report="$map/report.json";metrics=$report.route_metrics_summary;fingerprints=$fingerprints}
            }catch{
                @{map=$map;accepted=$false;error=$_.Exception.Message;report="$map/report.json"}
            }
        }
    }
    $jobs|Wait-Job|Out-Null
    foreach($job in $jobs){
        $result=@($job|Receive-Job)
        if($job.State -ne 'Completed' -or $result.Count -ne 1){
            $results+=@{map=$cases[$results.Count].map;accepted=$false;error='Regression worker did not produce a result'}
        }else{$results+=$result[0]}
    }
    $accepted=@($results|Where-Object {!$_.accepted}).Count -eq 0
    @{version=1;accepted=$accepted;timescale=2;parallel_servers=6;scope='independent_map_death_return_not_campaign_transition';results=$results}|ConvertTo-Json -Depth 12|Set-Content -LiteralPath (Join-Path $root 'report.json') -Encoding utf8
    if(!$accepted){throw "Return regression failed; see $root/report.json"}
    Write-Output "PASS: $root"
}finally{
    $jobs|Remove-Job -Force -ErrorAction SilentlyContinue
}
