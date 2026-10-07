[CmdletBinding()]
param([string]$Compiler='F:/src/quake2/buildenv/pkg/buildenv/mingw64/bin/gcc.exe')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$source=Get-Content (Join-Path (Split-Path $repo -Parent) 'yquake2/src/game/g_combat.c') -Raw
$match=[regex]::Match($source,'if \((g_test_monster_no_infighting->value[\s\S]*?)\)\s*\{\s*gi.dprintf\("g_test_monster_no_infighting blocked')
if(!$match.Success){throw 'Native fixture damage guard not found'}
$folder=Join-Path $repo 'workspace/build/no-infighting-test'
New-Item -ItemType Directory $folder -Force|Out-Null
$test=@'
#include <assert.h>
#include <string.h>
enum { SVF_MONSTER=2 };
typedef struct { int svflags; const char *classname; } entity;
typedef struct { int value; } cvar;
static cvar enabled={1},cheats={1},barrier={1},clients={1};
static cvar *g_test_monster_no_infighting=&enabled,*sv_cheats=&cheats,
 *g_test_combat_barrier=&barrier,*g_test_combat_clients=&clients;
static int blocked(entity *targ,entity *attacker) { return GUARD; }
int main(void) {
 entity alive={SVF_MONSTER,"monster_gunner"},dead={0,"monster_gunner"},
  parasite={SVF_MONSTER,"monster_parasite"},corpse={0,"monster_parasite"},
  player={0,"player"},world={0,"worldspawn"},unknown={0,0};
 assert(blocked(&parasite,&alive));
 /* Pending splash retains a dead Gunner as owner after SVF_MONSTER clears. */
 assert(blocked(&parasite,&dead));
 assert(!blocked(&corpse,&dead));
 assert(!blocked(&player,&dead));
 assert(!blocked(&parasite,&player));
 assert(!blocked(&parasite,&world));
 assert(!blocked(&parasite,&unknown));
 enabled.value=0; assert(!blocked(&parasite,&dead)); enabled.value=1;
 cheats.value=0; assert(!blocked(&parasite,&dead)); cheats.value=1;
 barrier.value=0; assert(!blocked(&parasite,&dead)); barrier.value=1;
 clients.value=2; assert(!blocked(&parasite,&dead));
 return 0;
}
'@
$test=$test.Replace('GUARD',$match.Groups[1].Value)
Set-Content "$folder/guard.c" $test
$savedPath=$env:PATH
try {
 $env:PATH=(Split-Path $Compiler -Parent)+';'+$savedPath
 & $Compiler -Wall -Wextra -Werror "$folder/guard.c" -o "$folder/guard.exe"
 if($LASTEXITCODE){throw 'Native guard regression compile failed'}
 & "$folder/guard.exe"
 if($LASTEXITCODE){throw 'Native guard regression failed'}
 Write-Output 'Native no-infighting guard: live/dead owners, corpse, player and fixture boundaries passed'
} finally {$env:PATH=$savedPath}
