function Read-ProjectileFixture([string]$Path) {
    $f=Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if($f.map -ne 'base1' -or $f.target_class -notin @('monster_flyer','monster_parasite')){throw 'Unsupported combat fixture map/class'}
    if(!$f.PSObject.Properties['light'] -or ($null -ne $f.light -and ($f.light -isnot [long] -and $f.light -isnot [int] -or $f.light -lt 0 -or $f.light -gt 255))){throw 'Fixture light must be null (BSP) or integer 0..255'}
    $coordinates=@('bot_origin','actor_origin','target_origin')
    if($f.PSObject.Properties['second_target_origin']){$coordinates+='second_target_origin'}
    if($f.PSObject.Properties['actor_walk_target']){
        $coordinates+='actor_walk_target'
        foreach($key in @('actor_walk_after_frames','actor_walk_frames')){
            if(!$f.PSObject.Properties[$key] -or ($f.$key -isnot [int] -and $f.$key -isnot [long]) -or $f.$key -lt 1 -or $f.$key -gt 1000){throw "Invalid fixture $key"}
        }
    } elseif($f.PSObject.Properties['actor_walk_after_frames'] -or $f.PSObject.Properties['actor_walk_frames']){throw 'Actor walking requires target'}
    foreach($key in $coordinates){
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
