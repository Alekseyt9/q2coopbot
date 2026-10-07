[CmdletBinding()]
param([Parameter(Mandatory)][string[]]$Models,
 [Parameter(Mandatory)][string[]]$Episodes,
 [Parameter(Mandatory)][string]$OutputRoot,
 [int[]]$Capacities=@(16,24,32),[int]$Port=34800,
 [string]$Registry='scripts/scenarios/combat-training/index.json')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$root=[IO.Path]::GetFullPath($(if([IO.Path]::IsPathRooted($OutputRoot)){$OutputRoot}else{Join-Path $repo $OutputRoot}))
if(Test-Path -LiteralPath $root){throw 'Fresh benchmark output required'}
if(!$Models.Count -or !$Episodes.Count -or @($Capacities|Where-Object {$_ -notin 4,8,16,24,32}).Count){throw 'Invalid benchmark workload'}
New-Item -ItemType Directory -Path $root|Out-Null
Push-Location $repo
try{
    $env:GOCACHE=Join-Path $repo 'workspace/build/gocache'
    & go build -o "$root/q2episode.exe" ./cmd/q2episode
    if($LASTEXITCODE){throw 'Episode compiler build failed'}
    $frozen=@()
    for($i=0;$i -lt $Models.Count;$i++){
        $source=(Resolve-Path -LiteralPath $Models[$i]).Path
        $weights=Get-Content $source -Raw|ConvertFrom-Json
        $weights.deterministic=$true
        $target="$root/model-$i.json"
        $weights|ConvertTo-Json -Depth 100|Set-Content $target -Encoding utf8NoBOM
        $frozen+=@{path=$target;source=$source;source_sha256=(Get-FileHash $source).Hash.ToLowerInvariant();sha256=(Get-FileHash $target).Hash.ToLowerInvariant()}
    }
    @{models=$frozen;episodes=$Episodes;split='validation';count=4;timescale=2;capacities=$Capacities}|ConvertTo-Json -Depth 8|Set-Content "$root/workload.json"
    $results=@()
    foreach($capacity in $Capacities){
        $plans=@()
        for($i=0;$i -lt $frozen.Count;$i++){
            $plan="$root/plan-$capacity-model-$i.json"
            & "$root/q2episode.exe" --registry $Registry --root $repo --episodes ($Episodes -join ',') --split validation --mode learned --model $frozen[$i].path --count 4 --out $plan --artifacts "$root/capacity-$capacity-model-$i"
            if($LASTEXITCODE){throw 'Benchmark plan failed'}
            $plans+=$plan
        }
        @{stage='capture';capacity=$capacity}|ConvertTo-Json|Set-Content "$root/progress.json"
        try{& "$PSScriptRoot/run_registered_combat_pool.ps1" -Plans $plans -MaxInstances $capacity -Port $Port -OutputRoot "$root/pool-$capacity"}
        catch{if(!(Test-Path "$root/pool-$capacity/report.json")){throw};Write-Warning "Capacity $capacity rejected; retaining failure evidence and testing the next capacity"}
        $report=Get-Content "$root/pool-$capacity/report.json" -Raw|ConvertFrom-Json
        $resources=@(Get-Content "$root/pool-$capacity/resources.jsonl"|ForEach-Object {$_|ConvertFrom-Json})
        $results+=@{capacity=$capacity;state=$report.state;captures=$report.usable_captures;seconds=$report.wall_seconds;game_frames=$report.actual_game_frames;frames_per_second=$report.aggregate_game_frames_per_second;cpu_mean_percent=($resources|Measure-Object system_cpu_percent -Average).Average;cpu_peak_percent=($resources|Measure-Object system_cpu_percent -Maximum).Maximum;minimum_free_memory_mb=($resources|Measure-Object available_memory_mb -Minimum).Minimum;peak_native_instances=($resources|Measure-Object active_native_instances -Maximum).Maximum;report="$root/pool-$capacity/report.json"}
        $results|ConvertTo-Json -Depth 8|Set-Content "$root/results.json"
    }
    $best=$results|Where-Object {$_.state -eq 'complete'}|Sort-Object frames_per_second -Descending|Select-Object -First 1
    @{state='complete';results=$results;recommended_instances=$best.capacity;scope='Identical frozen policies, scenes and validation seeds; scheduling benchmark, not independent policy selection'}|ConvertTo-Json -Depth 10|Set-Content "$root/report.json"
    @{stage='complete'}|ConvertTo-Json|Set-Content "$root/progress.json"
}catch{
    @{stage='failed';reason=$_.Exception.Message}|ConvertTo-Json|Set-Content "$root/progress.json"
    throw
}finally{Pop-Location}
