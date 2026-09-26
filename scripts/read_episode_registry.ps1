function Read-EpisodeRegistry([string]$Manifest) {
    $index = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
    if ($index.version -ne 2 -or !$index.files.Count) { throw 'Unsupported or empty episode manifest' }
    $folder = Split-Path -Parent $Manifest
    $seenFiles = @{}; $seenIDs = @{}; $episodes = @()
    foreach ($file in $index.files) {
        if ($file -notmatch '^[a-z0-9][a-z0-9-]*\.json$' -or $seenFiles.ContainsKey($file)) { throw "Invalid or duplicate episode file: $file" }
        $seenFiles[$file] = $true
        $episode = Get-Content -LiteralPath (Join-Path $folder $file) -Raw | ConvertFrom-Json
        if (!$episode.id -or $seenIDs.ContainsKey($episode.id)) { throw "Missing or duplicate episode ID in $file" }
        $seenIDs[$episode.id] = $true
        $episodes += $episode
    }
    foreach ($file in Get-ChildItem -LiteralPath $folder -Filter '*.json' -File) {
        if ($file.FullName -eq (Get-Item -LiteralPath $Manifest).FullName) { continue }
        if (!$seenFiles.ContainsKey($file.Name)) { throw "Episode missing from manifest: $($file.Name)" }
    }
    # Preserve the self-contained historical artifact format for replay/audit.
    return [pscustomobject]@{version=1;episodes=$episodes}
}
