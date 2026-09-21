# Q2 CoopBot

An experimental bot for cooperative playthroughs of the original
Quake II under [Yamagi Quake II](https://github.com/yquake2/yquake2).

The project is based on the source code from [Q2 Gladiator Bot Botlib Reconstruction](https://github.com/themuffinator/Q2-Gladiator-Bot).
It retains the original botlib v0.96 interface, AAS format, and Gladiator
botlib, but the current clean integration uses Yamagi's baseq2 game sources
as the gameplay baseline. The bot bridge is added through a small compatibility
layer and does not replace Yamagi's combat, collision, monster, or weapon code.

The former all-in-one Gladiator game module remains available as the legacy
`game` target for comparison. The clean port is built as `game_clean` while it
is being validated.

## What's Included

- the **sv coopbot <name> <skin> <charfile> <charname>** server command;
- passing the **coop** value to botlib;
- finding monsters outside the client slots;
- players are not treated as enemies in coop;
- basic compatibility with standard Quake II coop maps;
- Windows x86 and x64 builds.

This is not yet a complete autonomous companion for the entire campaign. The
bot can navigate using AAS, regroup through reachable elevators, inspect a
diagnostic map model, extract changelevel targets, remember local safe
positions, and follow the first blocker-driven objective HTN, but it does not
yet fully understand every level objective: scripted multi-step triggers and
transitions between maps still require further development.

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

cmake --build build-x86 --target gladiator game_clean --parallel 4
~~~

For x64, run the same commands with an x64 compiler and the `build-x64`
directory.

Output files:

~~~text
build-x86/src/game_clean/gamex86_clean.dll
build-x86/libgladiator.dll

build-x64/src/game/gamex86_64.dll
build-x64/libgladiator_x64.dll
~~~

## Installing in Yamagi

Create the mod directory:

~~~text
<Quake II>\YamagiQ2\coopbot\
~~~

For the current 32-bit Windows Yamagi build, copy the clean module and rename
it to the filename expected by the engine:

~~~text
gamex86_clean.dll -> coopbot_clean\game.dll
gamex86_clean.dll -> coopbot_clean\gamex86.dll
libgladiator.dll  -> coopbot_clean\gladiator.dll
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
- `coopbot_leash 1` — enables coop regroup toward the player, including
  independent elevator traversal; this is the default in coop, and can be
  set to `0` to disable it explicitly;
- `coopbot_soft_leash 384` and `coopbot_hard_leash 768` — distance thresholds
  for discouraging new encounters and forcing regroup;
- `coopbot_new_group_guard 1` — blocks non-urgent distant new encounters;
  `coopbot_enemy_group_radius 384` keeps nearby members of an active fight
  eligible.
- `coopbot_player_intent 1` — enables the heuristic player-intent model;
  `coopbot_intent_confidence 0.60` is the minimum confidence for tactical
  overrides, while `coopbot_intent_advance_speed 48` and
  `coopbot_intent_hold_speed 24` tune movement classification. A confident
  `ADVANCE`/`EXPLORE` permits a new group; a confident `RETREAT` interrupts
  an otherwise continuing battle chase.
- `coopbot_joint_retreat 1` keeps the bot in the cooperative retreat/cover
  decision for `coopbot_joint_retreat_duration 1.5` seconds after the player
  stops moving, preventing an immediate chase oscillation. The interval is
  logged as `coopbot_joint_retreat` with `phase=start/end`.
- `coopbot_roles 1` — enables temporary role modifiers and the bounded
  initiative budget; roles are logged as `coopbot_role` and currently cover
  `FOLLOWER`, `SUPPORT`, `ANCHOR`, `COVER`, `VANGUARD`, and `REGROUP`.
  Significant role changes also emit `coopbot_decision` records with the
  current AI node, player intent, initiative budget, target, confidence, and
  the selected reason, so a role transition can be reconstructed after play.
  `COVER`/`SUPPORT` also make a bounded lateral step when the bot occupies the
  player's direct line to its current enemy; `coopbot_role_position_interval`
  and `coopbot_role_position_radius` limit that positioning overlay.
- `coopbot_rescue 1` — enables `RESCUER` when the game supplies a confirmed
  critical-health/damage snapshot for the human. The rescue overlay closes a
  bounded gap toward the player, but the bot's own danger retreat and elevator
  regroup remain higher priority.
  The game-side bridge publishes this telemetry through namespaced internal
  libvars; the legacy export table is unchanged.
- `coopbot_shared_focus 1` — when the player is visibly firing at a monster,
  the bot scans that target first and retains the signal briefly for
  `coopbot_focus_memory 0.75` seconds after the last confirmed shot.
  `coopbot_kill_steal_control 1` yields a non-urgent focused target outside
  `coopbot_kill_steal_radius`, while a shooting or close threat still
  overrides the yield.
- `coopbot_intent_signal 1` briefly turns the bot toward its selected
  `COVER`/`RESCUER` movement before taking the step, making the intended
  maneuver readable without changing ordinary follow/avoidance movement.
- `coopbot_action_commitment 0.75` keeps a selected `COVER/SUPPORT` side for
  a short interval instead of recomputing left/right every positioning tick;
  the same commitment window also stabilizes `RESCUER`. A role change, hard
  regroup, or blocked move interrupts it. `coopbot_action` logs start/end
  for `POSITION` and `RESCUE`, and marks `REGROUP` transitions.
- `coopbot_target_hysteresis 1` — avoids replacing a current enemy for a
  marginally better candidate; `coopbot_target_switch_ratio 1.25` controls
  the required utility improvement.
- `coopbot_target_acquisition_delay 0.25` — keeps a newly noticed target
  pending for a short human-like reaction interval. Set it to `0` to disable;
  damage and actively shooting threats still interrupt immediately.
- `coopbot_burst_control 1` — enables short bursts for automatic coop
  weapons; `coopbot_burst_shots 4` and `coopbot_burst_pause 0.25` control
  burst length and the re-aim pause.
- `coopbot_danger_retreat 1` — enables the coop danger score; threshold
  defaults to `coopbot_danger_threshold 0.65`, with critical health at
  `coopbot_danger_critical_health 25`.
- `coopbot_basic_cover 1` — enables the P1 short-step cover fallback: when
  danger is high and the current enemy is visible, the bot tries a reachable
  backward or lateral step that breaks line of sight.
- `coopbot_safe_area_retreat 1` — enables the P5 safe-area fallback: areas
  observed without live enemies are remembered, and a dangerous retreat can
  return to the last reachable safe position when no higher-priority team or
  item retreat goal exists.
- `coopbot_elevator_wait_timeout 15` — resets a stuck elevator route after
  fifteen seconds of waiting.
- `coopbot_elevator_travel_timeout 30` — resets a route that selected an
  elevator but does not reach the target level within thirty seconds.
  During the lift ride, a temporarily unknown player AAS area uses the last
  valid area instead of cancelling vertical regroup.
- `coopbot_objective_htn 1` — enables the first map-objective slice: when a
  real navigation blocker resolves to a button/trigger, the bot navigates to
  that related control, activates it, and waits for the player before
  continuing. It does not scan or activate unrelated controls.
- `coopbot_objective_wait_distance 256` — distance at which the
  `OPEN_PATH/WAIT_FOR_PLAYER` step releases the bot.
- `coopbot_objective_wait_timeout 30` — maximum time in seconds to wait for
  the player after activating a control; expiry returns the objective to
  `RETRY` so the bot cannot remain paused forever.
- `coopbot_player_style 1` — enables the bounded EMA of player aggression,
  pace, preferred combat range, risk tolerance, retreat frequency, and
  exploration tendency. `coopbot_style_learning_rate 0.10` controls the EMA
  step and `coopbot_style_update_interval 1.0` limits updates to once per
  second. The model only adapts soft roles; regroup and rescue safety gates
  remain authoritative.
- `coopbot_advance_radius 256` — maximum player-to-bot distance for the
  area-transition gate to accept confident `ADVANCE`/`EXPLORE` intent.
- `coopbot_personal_space 1` — enables the opt-in close-range companion
  separation overlay; `coopbot_personal_space_radius 96` sets its radius.
- `coopbot_fireline_avoid 1` — lets the bot step sideways when it blocks the
  player's visible line to the bot's current enemy; radius defaults to 128.
- `coopbot_doorway_avoid 1` — lets the bot give way when the moving player
  crosses into its AAS area through a narrow passage; radius defaults to 160.
- `coopbot_map_model 1` — emits the initialized AAS area/reachability graph,
  elevator edges, and BSP control entities (`func_plat`, doors, buttons,
  triggers, changelevel transitions, and targets) once per map. During the episode it also records
  `coopbot_area_transition` for player/bot AAS-area changes, including
  `VISITED`, `ACTIVE_COMBAT`, `PARTIALLY_CLEARED`, `CLEARED`, `DANGEROUS`, or
  `REGROUP` observations. `coopbot_safe_area` records the last reachable
  no-enemy position. A bounded per-bot area ledger restores a room's
  remembered `VISITED`/combat/`CLEARED` state when the bot revisits it. With
  `coopbot_objective_htn 1`, a confident player `ADVANCE`/`EXPLORE` intent
  and `coopbot_advance_radius` release the area-transition gate; otherwise
  the companion waits instead of silently exploring. Control
  `target`/`targetname` links are emitted as `coopbot_map_control_link`; broken
  links are emitted as `coopbot_map_control_unresolved`. The parsed BSP
  control graph is cached for the loaded map and reused by the blocker-driven
  Objective HTN, so activation does not depend on reparsing a changing entity
  snapshot every frame.
  `coopbot_map_region` records a coarse room/sector segmentation derived from
  AAS clusters, including area count and aggregate bounds; portal areas remain
  boundary data rather than being assigned to both neighboring regions. The
  report tool can derive the same region records from `coopbot_map_area` lines
  when an older runtime logger does not emit the aggregate records directly.

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

Scenario 11 has report gates for repeatable checks:

~~~text
python tools/coopbot_event_report.py coopbot_debug_events.jsonl \
  --botlib-log botlib.log --require-elevator-edge --require-human-player
~~~

After a real two-client separation run, add
`--require-elevator-regroup --require-player-area-transition
--require-bot-area-transition`; these require a regroup frame whose selected
travel type is `TRAVEL_ELEVATOR` and evidence that both participants changed
AAS areas. A failed regroup emits
`coopbot_elevator_failed` plus `coopbot_path_failure phase=regroup`, so the
report distinguishes a failed elevator/path traversal from idle follow. The
`coopbot_elevator_boarded` and `coopbot_elevator_reacquired` records distinguish
boarding/travel from returning to the player's level; the latter can be required
with `--require-elevator-reacquired`.
`--require-vertical-elevator-edge` additionally requires the map model to show
different endpoint heights, which catches a missing or degenerate vertical
route before a runtime two-client test.
objective phases are also counted as `objective_regroup_approach`,
`objective_regroup_wait_elevator`, `objective_regroup_travel_elevator`,
`objective_regroup_retry`, and `objective_regroup_reacquire`. The
map-control objective emits `objective_open_path_navigate_control`,
`objective_open_path_activate`, `objective_open_path_wait_for_player`,
`objective_open_path_player_reacquired`, `objective_open_path_complete`, and
`objective_open_path_return_to_path`, and `objective_open_path_retry` when the
corresponding phases occur. `ACTIVATE` means the bot physically reached the
resolved control; `PLAYER_REACQUIRED` is the runtime confirmation used before
the objective is released. `ROUTE_CONFIRMED` is advisory AAS evidence that a
route from the bot's current area to the player's area exists after activation;
an unknown or transient player area is not treated as a failure.
`changelevel_gate_wait_for_player` and `changelevel_gate_release` record the
cooperative transition gate, which prevents a bot from touching a
changelevel-trigger volume while the player is still behind.
`--require-open-path-activation` requires reaching and activating the resolved
control; `--require-open-path-route` requires the advisory AAS route probe;
`--require-open-path-complete` additionally turns the completed
control objective into a report gate. `--require-safe-area` requires at least one remembered no-enemy
position. `--require-map-transition` requires at least one extracted
changelevel record in the map model. The
runtime activation check is separate: `--require-runtime-map-transition`
requires a `map_transition` event in the game-side JSONL log, proving that a
changelevel trigger actually fired during the run. The
negative-path check is available as `--require-regroup-path-failure`; successful
reacquisition emits `coopbot_regroup_complete` and can be required with
`--require-regroup-complete`.
The repeatable `--require-decision` gate accepts `role_select`, `target_select`,
`target_yield`, or `action_select`, so a baseline can require evidence that a
specific decision family occurred rather than treating map startup as a
successful companion run.
Each `episode_start` summary also retains its configured seed. After collecting
per-run reports, `tools/coopbot_baseline.py --min-runs 20 report-*.json`
checks that every report contains an episode seed and produces a machine-readable
aggregate; it deliberately fails until the requested number of real runs exists.
For cooperative acceptance runs, add `--require-human-player` to both the
report and baseline commands; this requires the explicit game-side
`player_is_human=1` marker, so a bot-only fallback slot cannot pass the gate.

When `--botlib-log` is supplied, `botlib_log.map_model` also includes the
concrete elevator edges, their source/destination AAS areas and bounds, BSP
control records, extracted map transitions, resolved control links, and
unresolved targets, plus coarse AAS-cluster regions and `region_connectors`
that identify elevator endpoint areas, clusters, and vertical displacement.
An AAS cluster is not assumed to be an individual floor: the endpoint heights
are retained because a real elevator can connect two levels inside one cluster.
This makes a map-level diagnosis inspectable without guessing from bot
movement alone.

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
