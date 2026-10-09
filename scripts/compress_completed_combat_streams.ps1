[CmdletBinding()]
param([Parameter(Mandatory)][string]$Root)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$boundary=[IO.Path]::GetFullPath((Join-Path $repo 'workspace/artifacts')).TrimEnd('\','/')+[IO.Path]::DirectorySeparatorChar
$target=(Resolve-Path -LiteralPath $Root).Path.TrimEnd('\','/')
if(!$target.StartsWith($boundary,[StringComparison]::OrdinalIgnoreCase)){throw 'Stream compression is limited to generated combat artifacts'}
for($parent=[IO.DirectoryInfo]::new($target);$parent -and $parent.FullName.StartsWith($boundary.TrimEnd('\'),[StringComparison]::OrdinalIgnoreCase);$parent=$parent.Parent){
 if($parent.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Stream compression refuses junction ancestors'}
}
$running=@(Get-CimInstance Win32_Process|Where-Object {
 ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($target+'\',[StringComparison]::OrdinalIgnoreCase)) -or
 ($_.Name -match '^(python.*|q2ppo-data|q2combat-export)\.exe$' -and $_.CommandLine -and $_.CommandLine.Contains($target))
})
if($running.Count){throw 'Stream writer/exporter still running'}
$log=Join-Path $target 'stream-compression.log'
& "$env:SystemRoot/System32/compact.exe" /C "/S:$target" /EXE:LZX /F /I /Q '*.jsonl' *> $log
if($LASTEXITCODE){throw "LZX stream compression failed: $log"}
@{version='combat_stream_storage_v1';encoding='utf8-jsonl';storage='windows-wof-lzx';transparent_reads=$true;scope='Closed JSONL streams only. Logical bytes, paths and content hashes retained; Windows decompresses normal reads.'}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $target 'stream-storage.json') -Encoding utf8NoBOM
