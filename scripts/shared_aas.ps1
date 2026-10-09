# AAS are immutable content-addressed files. Replace links, never shared bytes.
function Install-SharedAAS([string]$Source,[string]$Target) {
    $store=Join-Path (Split-Path $PSScriptRoot -Parent) 'workspace/build/aas-store'
    New-Item -ItemType Directory -Path $store -Force|Out-Null
    $digest=Get-RuntimeImmutableHash $Source
    $canonical=Join-Path $store "$digest.aas"
    if(!(Test-Path -LiteralPath $canonical)) {
        $pending=Join-Path $store ([guid]::NewGuid().ToString('N')+'.tmp')
        try {
            [IO.File]::Copy($Source,$pending)
            if((Get-RuntimeImmutableHash $pending) -ne $digest){throw 'AAS changed while snapshotting'}
            try{[IO.File]::Move($pending,$canonical)}catch{if(!(Test-Path -LiteralPath $canonical)){throw}}
        }finally{if(Test-Path -LiteralPath $pending){Remove-Item -LiteralPath $pending}}
    }
    if((Get-RuntimeImmutableHash $canonical) -ne $digest){throw 'Shared AAS content mismatch'}
    if(Test-Path -LiteralPath $Target) {
        $item=Get-Item -LiteralPath $Target
        if($item.LinkType -eq 'SymbolicLink' -and $item.Target -eq $canonical){return}
    }
    $temporary=Join-Path (Split-Path $Target -Parent) ('.a'+[guid]::NewGuid().ToString('N')+'.tmp')
    try {
        [IO.File]::CreateSymbolicLink($temporary,$canonical)|Out-Null
        [IO.File]::Move($temporary,$Target,$true)
    }finally{if(Test-Path -LiteralPath $temporary){Remove-Item -LiteralPath $temporary}}
}
