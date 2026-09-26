[CmdletBinding()]
param([Parameter(Mandatory)][string]$Config)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'harness_manifest.ps1')
$preparationTimer = [diagnostics.stopwatch]::StartNew()
$sourceRecords = @(Get-HarnessSourceRecords -Repository $repo)
$sourceFingerprint = Get-HarnessFingerprint -Records $sourceRecords
$configPath = (Resolve-Path -LiteralPath $Config).Path
$base = Split-Path -Parent $configPath
$cfg = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
foreach ($key in $cfg.PSObject.Properties.Name) {
    if ($key -notin @('version','scenarios','sessions','timescales','repetitions','parallelism','base_port','runtime_root','tail_frames')) { throw "Unknown suite field: $key" }
}
$definitions=@()
foreach($path in $cfg.scenarios) {if($path) {$definitions+=@{path=$path;kind='scenario'}}}
foreach($path in $cfg.sessions) {if($path) {$definitions+=@{path=$path;kind='session'}}}
if ($cfg.version -ne 1 -or $cfg.parallelism -lt 1 -or $cfg.parallelism -gt 8 -or $cfg.repetitions -lt 1 -or $cfg.repetitions -gt 100 -or $definitions.Count -lt 1 -or @($cfg.timescales).Count -lt 1) { throw 'Invalid suite configuration.' }
if(@($definitions | Where-Object kind -eq 'session').Count -gt 0 -and $cfg.tail_frames -lt 2) {throw 'Session suites require tail_frames >= 2'}
foreach ($scale in $cfg.timescales) { if ($scale -notin @(1,2)) { throw 'Only validated timescales 1 and 2 are supported.' } }
$sourceRuntime = (Resolve-Path -LiteralPath (Join-Path $base $cfg.runtime_root)).Path
$output = Join-Path $repo ('workspace/artifacts/scenario-suite-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $output | Out-Null
Copy-Item -LiteralPath $configPath -Destination (Join-Path $output 'suite-config.json')
$client = Join-Path $output 'q2coopbot.exe'
$reporter = Join-Path $output 'q2scenario-report.exe'
Push-Location $repo
try {
    & go build -o $client ./cmd/q2coopbot
    if ($LASTEXITCODE) { throw 'Client build failed.' }
    & go build -o $reporter ./cmd/q2scenario-report
    if ($LASTEXITCODE) { throw 'Reporter build failed.' }
} finally { Pop-Location }
$tasks = @()
if ($null -ne $cfg.tail_frames -and $cfg.tail_frames -ne 0 -and ($cfg.tail_frames -lt 2 -or $cfg.tail_frames -gt 1000)) {throw 'tail_frames must be zero or 2..1000.'}
foreach ($definition in $definitions) {
    $scenario = (Resolve-Path -LiteralPath (Join-Path $base $definition.path)).Path
    foreach ($scale in $cfg.timescales) {
        for ($repeat = 1; $repeat -le $cfg.repetitions; $repeat++) {
            $port = [int]$cfg.base_port + $tasks.Count
            if ($port -lt 1024 -or $port -gt 65534) { throw 'Invalid port range.' }
            if (Get-NetUDPEndpoint -LocalPort $port -ErrorAction SilentlyContinue) { throw "UDP port already occupied: $port" }
            $dir = Join-Path $output ('run-{0:D3}' -f $tasks.Count)
            $runtime = Join-Path $dir 'runtime'
            New-Item -ItemType Directory -Path (Join-Path $runtime 'baseq2/maps') -Force | Out-Null
            # Executables and mutable runtime state are private. Only read-only assets are linked.
            Get-ChildItem -LiteralPath $sourceRuntime -File | Where-Object { $_.Extension -in @('.exe','.dll') } | Copy-Item -Destination $runtime
            Copy-Item -LiteralPath (Join-Path $sourceRuntime 'baseq2/game.dll') -Destination (Join-Path $runtime 'baseq2/game.dll')
            $assetRoot = Join-Path $sourceRuntime 'baseq2'
            foreach ($asset in Get-ChildItem -LiteralPath $assetRoot -File -Recurse | Where-Object { $_.Extension -in @('.pak','.bsp','.aas') }) {
                $relative = $asset.FullName.Substring($assetRoot.Length+1)
                $dest = Join-Path (Join-Path $runtime 'baseq2') $relative
                New-Item -ItemType Directory -Path (Split-Path -Parent $dest) -Force | Out-Null
                try { New-Item -ItemType HardLink -Path $dest -Target $asset.FullName -ErrorAction Stop | Out-Null }
                catch { Copy-Item -LiteralPath $asset.FullName -Destination $dest }
            }
            $snapshot = Join-Path $dir 'scenario.json'
            Copy-Item -LiteralPath $scenario -Destination $snapshot
            $sessionConfig=$null
            if($definition.kind -eq 'session') {
                $sessionConfig=Join-Path $dir 'session-trial-config.json'
                @{session=$snapshot;runtime_root=$runtime;port=$port;timescale=$scale;tail_frames=[int]$cfg.tail_frames} | ConvertTo-Json | Set-Content $sessionConfig -Encoding utf8
            }
            $tasks += [pscustomobject]@{ kind=$definition.kind; sessionConfig=$sessionConfig; scenario=$snapshot; scale=$scale; repeat=$repeat; port=$port; dir=$dir; runtime=$runtime; tail=[int]$cfg.tail_frames }
        }
    }
}
$worker = {
    param($task,$runner,$client,$reporter)
    $ErrorActionPreference = 'Stop'
    $result = [ordered]@{ state='infrastructure_failed'; accepted=$false; port=$task.port; timescale=$task.scale; repeat=$task.repeat; directory=$task.dir; fixture_fingerprint=$task.fixtureFingerprint; started_utc=[datetime]::UtcNow.ToString('o') }
    $result.kind=$task.kind
    try {
        if($task.kind -eq 'session') {
            $sessionRunner=Join-Path (Split-Path -Parent $runner) 'run_session_trial.ps1'
            & $sessionRunner -Config $task.sessionConfig -PreparedOutput $task.dir -ClientExe $client -ReporterExe $reporter -ReturnRejectedReport *> (Join-Path $task.dir 'runner.log')
            $reportPath=Join-Path $task.dir 'report.json'
            $report=Get-Content $reportPath -Raw | ConvertFrom-Json
            $result.state=$report.state
            $result.accepted=$report.accepted
            $result.reason=$report.reason
            $result.problem_location=$report.problem_location
            $result.expectation='normal_completion'
            $phaseMetrics=[ordered]@{}
            for($phase=0;$phase -lt $report.phases.Count;$phase++) {$phaseMetrics["phase_$phase"]=$report.phases[$phase].report.metrics}
            $result.metrics=@{phases=$phaseMetrics;observer_commands=$report.observer_commands}
            $result.speed=Get-Content (Join-Path $task.dir 'session-timing.json') -Raw | ConvertFrom-Json
            $result.report=$reportPath
        } else {
        $s = Get-Content -LiteralPath $task.scenario -Raw | ConvertFrom-Json
        $args = @{ RuntimeRoot=$task.runtime; ClientExe=$client; ActorScenario=$task.scenario; GameFrames=$s.game_frames; Timescales=@($task.scale); Port=$task.port; OutputRoot=$task.dir; SynchronizedStart=$true; UnlimitedLoopbackRate=$true }
        if ($s.map -ne 'base1') { $args.TransitionMap=$s.map; $args.TransitionAfterFrames=10 }
        if ($task.tail) {$args.ScenarioTailFrames=$task.tail}
        & $runner @args *> (Join-Path $task.dir 'runner.log')
        $summary = Get-Content -LiteralPath (Join-Path $task.dir 'summary.json') -Raw | ConvertFrom-Json
        $reportPath = Join-Path $task.dir 'report.json'
        $reportConfig = Join-Path $task.dir 'report-config.json'
        @{scenario=$task.scenario; actor_trace=$summary.human_trace_jsonl; bot_trace=$summary.trace_jsonl; output=$reportPath} | ConvertTo-Json | Set-Content -LiteralPath $reportConfig -Encoding utf8
        & $reporter --config $reportConfig *> (Join-Path $task.dir 'reporter.log')
        $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        $result.state=$report.state
        $result.accepted=$report.accepted
        $result.expectation=$report.expectation
        $result.metrics=$report.metrics
        $result.early_stop=$summary.scenario_early_stop
        $result.speed=[ordered]@{ wall_seconds=$summary.wall_seconds; game_fps=$summary.game_fps; measured_speedup=([double]$summary.game_fps/10); frame_gaps=$summary.frame_gaps; decode_errors=$summary.decode_errors; sent_commands=$summary.sent_commands; matched_applied_commands=$summary.matched_applied_commands }
        $result.report=$reportPath
        }
    } catch { $result.state='infrastructure_failed'; $result.accepted=$false; $result.error=$_.Exception.Message }
    $result.finished_utc=[datetime]::UtcNow.ToString('o')
    $result | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $task.dir 'result.json') -Encoding utf8
    [pscustomobject]$result
}
$artifactPaths = @($client,$reporter,(Join-Path $output 'suite-config.json'))
foreach ($task in $tasks) {
    $artifactPaths += $task.scenario
    if($task.sessionConfig) {$artifactPaths += $task.sessionConfig}
    $artifactPaths += @(Get-ChildItem -LiteralPath $task.runtime -Recurse -File | ForEach-Object FullName)
}
$artifactRecords = @(Get-HarnessFileRecords -Root $output -Paths $artifactPaths)
foreach ($task in $tasks) {
    $relativeRun = [IO.Path]::GetFileName($task.dir)
    $runtimePrefix=$relativeRun+'/runtime/'
    $fixtureRecords=@($artifactRecords | Where-Object { $_.path.StartsWith($runtimePrefix) } | ForEach-Object { [pscustomobject]@{path=$_.path.Substring($runtimePrefix.Length);sha256=$_.sha256} })
    $scenarioHash=($artifactRecords | Where-Object path -eq ($relativeRun+'/scenario.json')).sha256
    $fixtureRecords+=@([pscustomobject]@{path='scenario';sha256=$scenarioHash},[pscustomobject]@{path='timescale';sha256=[string]$task.scale},[pscustomobject]@{path='tail_frames';sha256=[string]$task.tail})
    if($task.kind -eq 'session') {$fixtureRecords+=@([pscustomobject]@{path='kind';sha256='session'})}
    $task | Add-Member fixtureFingerprint (Get-HarnessFingerprint -Records $fixtureRecords)
}
$gitRevision = & git -C $repo rev-parse HEAD
if ($LASTEXITCODE) {throw 'Cannot record Git revision.'}
$gitStatus = @(& git -C $repo status --porcelain)
if ($LASTEXITCODE) {throw 'Cannot record Git status.'}
$goVersion = & go version
$manifest = [ordered]@{
    version=1; created_utc=[datetime]::UtcNow.ToString('o'); revision=$gitRevision; dirty=($gitStatus.Count -gt 0)
    go_version=$goVersion; powershell_version=$PSVersionTable.PSVersion.ToString()
    source_fingerprint=$sourceFingerprint; source_files=$sourceRecords; artifact_files=$artifactRecords
    server_seed=$null; deterministic_world_verified=$false; source_unchanged=$null; artifacts_unchanged=$null
}
$manifestPath=Join-Path $output 'manifest.json'
$manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $manifestPath -Encoding utf8
$preparationTimer.Stop()
$jobs = @()
$results = @()
$next = 0
$timer = [diagnostics.stopwatch]::StartNew()
try {
    while ($next -lt $tasks.Count -or $jobs.Count) {
        while ($next -lt $tasks.Count -and $jobs.Count -lt $cfg.parallelism) {
            $jobs += Start-Job -ScriptBlock $worker -ArgumentList $tasks[$next],(Join-Path $PSScriptRoot 'run_speed_trial.ps1'),$client,$reporter
            $next++
        }
        $done = Wait-Job -Job $jobs -Any -Timeout 1
        if ($done) {
            $received = @(Receive-Job -Job $done -ErrorAction Continue)
            if (-not $received.Count) { $results += [pscustomobject]@{state='infrastructure_failed'; error='Worker exited without result'} }
            else { $results += $received }
            $jobs = @($jobs | Where-Object Id -ne $done.Id)
            Remove-Job -Job $done
        }
    }
} finally {
    foreach ($job in $jobs) { Stop-Job -Job $job; Remove-Job -Job $job }
}
$timer.Stop()
$verificationTimer=[diagnostics.stopwatch]::StartNew()
$manifest.source_unchanged = (Get-HarnessFingerprint -Records @(Get-HarnessSourceRecords -Repository $repo)) -eq $sourceFingerprint
$manifest.artifacts_unchanged = (Get-HarnessFingerprint -Records @(Get-HarnessFileRecords -Root $output -Paths $artifactPaths)) -eq (Get-HarnessFingerprint -Records $artifactRecords)
$manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $manifestPath -Encoding utf8
$verificationTimer.Stop()
$passed = @($results | Where-Object accepted -eq $true).Count
$expectedFailures = @($results | Where-Object expectation -eq 'expected_failure_matched').Count
$fps = @($results | Where-Object {$null -ne $_.speed.game_fps} | ForEach-Object { $_.speed.game_fps })
$stats = $fps | Measure-Object -Minimum -Maximum -Average
$summary = [ordered]@{ runs=$tasks.Count; passed=$passed; expected_failures=$expectedFailures; failed=($tasks.Count-$passed); parallelism=$cfg.parallelism; execution_wall_seconds=$timer.Elapsed.TotalSeconds; episodes_per_minute=($tasks.Count*60/$timer.Elapsed.TotalSeconds); game_fps_min=$stats.Minimum; game_fps_max=$stats.Maximum; game_fps_mean=$stats.Average; results=$results }
$summary.manifest=$manifestPath
$summary.provenance_valid=($manifest.source_unchanged -and $manifest.artifacts_unchanged)
$summary.preparation_seconds=$preparationTimer.Elapsed.TotalSeconds
$summary.verification_seconds=$verificationTimer.Elapsed.TotalSeconds
$summary | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath (Join-Path $output 'suite-summary.json') -Encoding utf8
Write-Output "Suite: $output ($passed/$($tasks.Count) passed)"
if ($passed -ne $tasks.Count) { throw 'Scenario suite failed; see suite-summary.json.' }
if (-not $summary.provenance_valid) {throw 'Scenario suite inputs changed while running; see manifest.json.'}
