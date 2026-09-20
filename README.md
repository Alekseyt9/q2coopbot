# Q2 CoopBot

An experimental bot for cooperative playthroughs of the original
Quake II under [Yamagi Quake II](https://github.com/yquake2/yquake2).

The project is based on the source code from [Q2 Gladiator Bot Botlib Reconstruction](https://github.com/themuffinator/Q2-Gladiator-Bot).
It retains the original botlib v0.96 interface, AAS format, and most of the
original Gladiator game module. This repository adds the changes required for
the first coop MVP.

## What's Included

- the **sv coopbot <name> <skin> <charfile> <charname>** server command;
- passing the **coop** value to botlib;
- finding monsters outside the client slots;
- players are not treated as enemies in coop;
- basic compatibility with standard Quake II coop maps;
- Windows x86 and x64 builds.

This is not yet a complete autonomous companion for the entire campaign. The
bot can navigate using AAS and find and attack monsters, but it does not yet
fully understand level objectives: buttons, doors, elevators, scripted
triggers, and transitions between maps require further development.

## Requirements

- original Quake II files from a legitimate installation;
- Yamagi Quake II for Windows;
- CMake and Ninja;
- MinGW-w64 or another compatible C compiler.

The official Yamagi Windows package usually runs as **i386**, so the x86
CoopBot build should be used with it. This project also builds x64 DLLs for
x64 engine builds.

## Building

MinGW-w64 examples from PowerShell:

~~~powershell
cmake -S . -B build-x86 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++

cmake --build build-x86 --target gladiator game --parallel 4
~~~

For x64, run the same commands with an x64 compiler and the `build-x64`
directory.

Output files:

~~~text
build-x86/src/game/gamex86.dll
build-x86/libgladiator.dll

build-x64/src/game/gamex86_64.dll
build-x64/libgladiator_x64.dll
~~~

## Installing in Yamagi

Create the mod directory:

~~~text
<Quake II>\YamagiQ2\coopbot\
~~~

For the current 32-bit Windows Yamagi build, copy:

~~~text
gamex86.dll       -> coopbot\game.dll
libgladiator.dll  -> coopbot\gladiator.dll
~~~

The mod directory must also contain the Gladiator data:

~~~text
coopbot\pak7.pak
coopbot\bots.cfg
coopbot\Gladiator.gsl
coopbot\default\
~~~

These files come from the original Gladiator distribution or from the archived
copy in `archive/gladiator-bot/`. The Quake II files `pak0.pak`, `pak1.pak`, and
`pak2.pak` are not included in this repository.

If MinGW built a DLL with a dynamic dependency on libgcc, place
`libgcc_s_dw2-1.dll` next to `q2ded.exe` or `quake2.exe`.

## Starting Coop with a Bot

Start Yamagi with the mod:

~~~text
quake2.exe -portable +set game coopbot
~~~

Run the following commands in the game console:

~~~text
map base1
exec coopbot.cfg
~~~

The `coopbot.cfg` file contains:

~~~text
sv coopbot "RangerBot" "male/grunt" "bots/player_c.c" "player"
~~~

To add another bot manually, use the same command with a different name, skin,
and character file. Use the `status` command on the server to verify the result.

## Diagnostics and Metrics

The game module contains opt-in diagnostics for collecting data during bot
testing. Load the ready-made profile from the server console:

~~~text
exec coopbot_debug.cfg
~~~

The profile enables:

- `coopbot_log 2` — lifecycle events, map loading, bot input changes, weapon
  trace results, damage attempts, applied damage, and detailed botlib
  target/node events;
- `coopbot_metrics 1` — periodic aggregate reports;
- `coopbot_metrics_interval 10` — report interval in seconds;
- `coopbot_slow_ai_ms 100` — warning threshold for a slow bot AI call.

Log levels are cumulative: `0` disables CoopBot logs, `1` keeps lifecycle,
error, and slow-AI messages, `2` adds input and botlib target/node events, and
`3` also records every AI call. The `coopbot_metrics` console command prints a
report immediately.

CoopBot diagnostics are written to `coopbot_debug.log` inside the active mod
directory, so shot and damage traces do not fill the game console. The
timestamped botlib events are written to `botlib.log` when the botlib logger is
enabled. Useful records include `shot`, `damage_attempt`, `damage_applied`,
`event=enemy`, `event=node`, and `event=ai`.
The timing fields use process CPU time (`clock()`), so they are intended for
relative comparisons between bot versions rather than wall-clock profiling.

For a useful test sample, start one bot on a known map, play until it gets
stuck or fails to fight, then save both logs together with the map name and
the exact bot configuration. This makes it possible to correlate navigation,
target selection, input decisions, and performance regressions.

## AAS for New Maps

Gladiator requires a `.aas` navigation file. For a map that is not included in
the existing data, extract its BSP and run BSPC:

~~~text
bspc.exe -bsp2aas path\to\map.bsp -output path\to\coopbot
~~~

The result should be located here:

~~~text
coopbot\maps\map.aas
~~~

The working installation already includes `coopbot\maps\base1.aas` for base1.

## Origins and Acknowledgements

The project is primarily based on [Q2-Gladiator-Bot](https://github.com/themuffinator/Q2-Gladiator-Bot),
a Gladiator botlib reconstruction repository. The game module is derived from
the published Gladiator Bot source code; the reconstructed botlib reproduces
the original library's interface and behavior as closely as possible based on
the available materials.

Special thanks to Mr Elusive / Jan Paul van Waveren for the original Gladiator
Bot and the AAS architecture.

Q2 CoopBot is not affiliated with id Software or Yamagi Software.

## Status

This is an experimental project. It has been tested with Yamagi Q2 8.70a, the
original base1 map, and a 32-bit Windows build. The changes inherit the
limitations and warnings of the original Gladiator project.
