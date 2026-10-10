# Supervisor-only cursor: consume complete appended rows without losing death frames.
function New-CombatTraceCursor([string]$Path) {
    @{path=$Path;offset=0L;pending=[byte[]]@()}
}

function Read-CombatTraceRows($Cursor,[ValidateRange(1,8388608)][int]$MaxReadBytes=1048576) {
    if(!(Test-Path -LiteralPath $Cursor.path)){return}
    $file=[IO.File]::Open($Cursor.path,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::ReadWrite)
    try {
        if($file.Length -lt $Cursor.offset){throw 'Combat trace truncated during supervision'}
        $available=$file.Length-$Cursor.offset
        if($available -le 0){return}
        $length=[int][math]::Min($available,$MaxReadBytes)
        $buffer=[byte[]]::new($length)
        $null=$file.Seek($Cursor.offset,[IO.SeekOrigin]::Begin)
        $count=$file.Read($buffer,0,$length)
        $combined=[byte[]]::new($Cursor.pending.Length+$count)
        [Array]::Copy($Cursor.pending,0,$combined,0,$Cursor.pending.Length)
        [Array]::Copy($buffer,0,$combined,$Cursor.pending.Length,$count)
        $Cursor.offset+=$count
        $last=[Array]::LastIndexOf[byte]($combined,[byte]10)
        if($last -lt 0){
            if($combined.Length -gt 4MB){throw 'Combat trace line exceeds supervision limit'}
            $Cursor.pending=$combined
            return
        }
        $remaining=$combined.Length-$last-1
        if($remaining -gt 4MB){throw 'Combat trace line exceeds supervision limit'}
        $Cursor.pending=[byte[]]::new($remaining)
        if($remaining){[Array]::Copy($combined,$last+1,$Cursor.pending,0,$remaining)}
        # Decode only complete lines; incomplete UTF-8 bytes remain unmodified.
        $text=[Text.UTF8Encoding]::new($false,$true).GetString($combined,0,$last+1)
        foreach($line in $text.Split([char]10)){
            if($line.Trim()){$line|ConvertFrom-Json -ErrorAction Stop}
        }
    } finally {$file.Dispose()}
}
