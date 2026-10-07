[CmdletBinding()]
param([int]$Port,[int]$Count,[string]$Output,[string]$StopFile)
$ErrorActionPreference='Stop'
while(!(Test-Path -LiteralPath $StopFile)){
    $cpu=Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'"
    $mem=Get-CimInstance Win32_OperatingSystem
    $native=@(Get-NetUDPEndpoint -ErrorAction SilentlyContinue|Where-Object {$_.LocalPort -ge $Port -and $_.LocalPort -lt $Port+$Count}|Select-Object -ExpandProperty OwningProcess -Unique)
    @{utc=[DateTime]::UtcNow.ToString('o');system_cpu_percent=$cpu.PercentProcessorTime;available_memory_mb=[math]::Round($mem.FreePhysicalMemory/1024);active_native_instances=$native.Count}|ConvertTo-Json -Compress|Add-Content -LiteralPath $Output -Encoding utf8NoBOM
    Start-Sleep -Milliseconds 1000
}
