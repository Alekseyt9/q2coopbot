function Read-ProjectileFixture([string]$Path) {
    $f=Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if($f.map -ne 'base1' -or $f.target_class -ne 'monster_flyer'){throw 'Unsupported projectile fixture map/class'}
    foreach($key in @('bot_origin','actor_origin','target_origin')){
        if(@($f.$key).Count -ne 3){throw "Invalid fixture $key"}
        foreach($v in $f.$key){
            if($v -is [string] -or $v -is [bool] -or $null -eq $v -or [double]::IsNaN([double]$v) -or [double]::IsInfinity([double]$v) -or [math]::Abs([double]$v) -gt 8192){throw "Invalid fixture coordinate $key"}
        }
    }
    $f
}
function Format-ProjectileOrigin($Origin) {
    ($Origin|ForEach-Object {([double]$_).ToString([cultureinfo]::InvariantCulture)}) -join ','
}
