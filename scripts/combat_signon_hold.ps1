function Wait-CombatSignonHold {
    param([Parameter(Mandatory)]$Server,
          [Parameter(Mandatory)][string]$ServerLog,
          [ValidateRange(1,99)][int]$Frame=40)
    # UDP readiness alone permits signon at different world frames. Wait for
    # the existing native hold before connecting, so fixture entity allocation
    # cannot depend on client load time. This does not advance the held world.
    $deadline=[DateTime]::UtcNow.AddSeconds(15)
    do {
        if($Server.HasExited){throw 'Server exited before fixed signon hold'}
        if(Test-Path -LiteralPath $ServerLog){
            $matches=@(Get-Content -LiteralPath $ServerLog | Select-String '^sv_test_signon_hold spawncount=(-?\d+) frame=(\d+)$')
            if($matches.Count){
                if($matches.Count -ne 1 -or [int]$matches[0].Matches[0].Groups[2].Value -ne $Frame){throw 'Unexpected native signon hold receipt'}
                return $Frame
            }
        }
        if([DateTime]::UtcNow -ge $deadline){throw 'Native signon hold receipt timed out'}
        Start-Sleep -Milliseconds 50
    }while($true)
}
