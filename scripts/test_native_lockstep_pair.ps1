[CmdletBinding()]
param([string]$Compiler='F:/src/quake2/buildenv/pkg/buildenv/mingw64/bin/gcc.exe')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$native=Join-Path (Split-Path $repo -Parent) 'yquake2/src/server/header'
$folder=Join-Path $repo 'workspace/build/lockstep-pair-test'
New-Item -ItemType Directory $folder -Force|Out-Null
$source=@'
#include <assert.h>
#include <string.h>
#include "test_lockstep_pair.h"
int main(void) {
 int first;
 for(first=0;first<2;first++) {
  test_lockstep_pair_t p={0};
  assert(TestLockstepPairAccept(&p,first,first+1,101+first));
  assert(p.mask!=3);
  assert(!TestLockstepPairAccept(&p,first,first+1,202));
  assert(!TestLockstepPairAccept(&p,1-first,first+1,203));
  assert(TestLockstepPairAccept(&p,1-first,2-first,102-first));
  assert(p.mask==3 && p.actor[0]==1 && p.actor[1]==2);
  assert(p.sequence[0]==101 && p.sequence[1]==102);
  assert(!TestLockstepPairAccept(&p,0,1,999));
  memset(&p,0,sizeof(p));
  assert(TestLockstepPairAccept(&p,0,1,103));
 }
 { test_lockstep_pair_t p={0};
  assert(!TestLockstepPairAccept(&p,-1,1,1));
  assert(!TestLockstepPairAccept(&p,2,1,1));
  assert(!TestLockstepPairAccept(&p,0,0,1));
  assert(p.mask==0);
 }
 return 0;
}
'@
$source|Set-Content -LiteralPath "$folder/pair.c" -Encoding ascii
$saved=$env:PATH
try{
 $env:PATH=(Split-Path $Compiler -Parent)+';'+$saved
 & $Compiler -Wall -Wextra -Werror -I $native "$folder/pair.c" -o "$folder/pair.exe"
 if($LASTEXITCODE){throw 'Pair regression compilation failed'}
 & "$folder/pair.exe"
 if($LASTEXITCODE){throw 'Pair regression failed'}
 'Pair accept: both arrival orders, duplicates, duplicate actor, invalid role and reset passed'
}finally{$env:PATH=$saved}
