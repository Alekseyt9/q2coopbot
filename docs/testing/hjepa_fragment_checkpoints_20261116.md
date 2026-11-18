# H-JEPA short native fragments: checkpoint foundation

User-approved direction: replay short2–5second native combat continuations from
saved world states, then train/evaluate complete combat. Existing U1 remains
unchanged. This document distinguishes proven components from the unimplemented
end-to-end combat branch; it is not a completed fragment-training claim.

## Implemented policy components

`internal/policy/hjepa_checkpoint.go` adds a versioned sampling checkpoint for
frame-wise sampled BC H-JEPA. It binds the exact loaded model and original
logical sampling identity to a fresh native identity after load/reconnect.
Both primitive Gaussian/fire/pose and categorical weapon draws use that logical
clock; observations/actions retain their actual native identity. Default models
have no mapping and keep their old behavior. Reset clears the mapping. Wrong
model, map, actor, life transition or clock range is rejected; held weapon
options and other planners are not silently admitted. Nested checkpoints retain
the original logical origin.

`observation_checkpoint.go` captures only past client observation memory:
four-frame history, track counter and past observed threats. JSON snapshots are
deep copies. Restore rebases observation ages across a native clock reset,
including negative historical frame numbers. It does not restore current health,
inventory or hidden entity positions; those must come from the native save.

Focused tests prove:

- 64subsequent primitive and stochastic-weapon draw pairs survive a connection,
 generation and frame-clock change, with equal likelihoods and native action IDs.
- 37future observation steps retain all871feature values and actions/samples,
 including visible projectiles, occlusion, defeated targets and new track IDs.
- Corrupted temporal/track state, model/actor/map mismatches and aliasing are rejected.

`go test ./...` passed after these Go changes. These components are not yet
wired into the client's native combat capture/load lifecycle.

## Native save/load infrastructure verification

The first existing checkpoint branch test failed before save because shared
`.aas` paths are symbolic links. The old trial created a hard link to that
reparse point, while checkpoint runtime provenance requires regular files.
`run_checkpoint_trial.ps1` now materializes private AAS bytes through streams;
the strict checkpoint file validation remains intact. Failure evidence:
`workspace/artifacts/checkpoint-branches-20261116-203827-378/seed/save.err`.

Unseeded native test passed with an explicit load barrier:
`checkpoint-branches-20261116-204033-041/report.json`.
It verifies two independent native processes, copied package equality,
source immutability, mutation isolation and resumed scenario progress. It did
not enable the engine's harness RNG serialization and therefore did not prove
RNG continuation.

The branch runner now accepts optional `-Seed` (alias for `EngineSeed`), carries
it into each isolated process before map initialization, and requires the native
`rng_restored` acknowledgement on every seeded load. The first seed attempt
exposed PowerShell's case-insensitive conflict with the existing local `$seed`
report variable; the parameter was renamed. Its failed evidence is retained in
`checkpoint-branches-20261116-204417-486`.

`scripts/test_native_rng.ps1` passed: identical/different seeds, legacy zero,
exact saved-cursor continuation and truncated-save rejection.
The corrected seeded suite also passed:
`checkpoint-branches-20261116-204700-502/report.json`.
Both branch load receipts report `rng_restored=true`, `state=load_held`, new
generations and native frame1. Package copies match and modifications stay
isolated. Seed1729,2×speed, two branch processes.

This native suite uses the existing grounded, noncombat scenario. Active
projectile/grenade state and H-JEPA's combined native continuation remain
unverified. A save/load acknowledgement or RNG receipt alone is insufficient.

## Required next integration

1. Add an explicit versioned combat path, keeping legacy idle-planner capture
   restrictions. The current path rejects airborne states, active projectiles
   and hand-grenade preparation and omits `combatControl`.
2. Capture at a clearly defined **pre-decision native boundary**. The existing
   callback waits until the frame's command was sent, while the native save
   barrier rejects commands at/after its anchor. A combat implementation must
   not lose or double-apply that pending command. Use an explicit coordinated
   release acknowledgement, not a wall-clock delay.
3. Carry observation history/threats plus engagement bookkeeping, previous
   applied command, own weapon-request history, life/health bookkeeping and
   logical sampler origin. Bind all payloads to model/package/world anchors.
   For a pre-decision load at native frame1, the last observed-memory anchor
   ends at frame0. The explicitly named before-decision API now supports that
   internal past-memory anchor; ordinary restore/action roots remain positive.
4. Restore on the fresh native snapshot before policy/history update. Suppress
   fixture placement/reinitialization and re-establish synchronous stepping at
   the new network clock; preserve actual native entity/weapon/timer state.
5. First compare identical-policy continuations: observations,871features,
   proposals, guarded/native commands and damage/projectile/weapon events.
   Test both an ordinary shot and an already primed grenade. Examine allocator
   high-water/free-edict timing too: stock saves serialize in-use entities;
   exact stochastic gameplay after load is not established by the quiet suite.
6. Only after that proof, fork policy sampling streams for2–5second fragments,
   retain behavior probabilities and native outcomes, and admit a distinct
   fragment dataset type. Existing full-episode PPO export must not accidentally
   ingest it. Do not treat forced diagnostic commands as sampled policy labels.

Fragments ending before terminal effects require longer continuation or a
frozen state-value estimate. A cutoff is truncation, not a death or a zero-value
terminal. Source checkpoint families, not repeated branches, are the independent
world units; keep TRAIN and fresh evaluation banks separate. Full battles
remain necessary for quality and transfer checks. No training or promotion has
occurred in this checkpoint-foundation stage.

Live U1 SHA256 remains
`21c3b680e0b6090d601d524f6da53d73ae6dd083c8e7a9a6da2e6d268ccadd26`.


## Pre-decision controller memory, 21:04 continuation

Implemented `internal/bot/combat_checkpoint.go` with the distinct
`hjepa_predecision_checkpoint_v1` payload. It captures observation history,
threat memory, sampled H-JEPA logical clock, preceding sent command, life/HP,
own weapon requests, observed engagement identities/last-seen frame and previous
target intent. It binds execution contract and guard settings, rejects diagnostic
and campaign modes, requires a fresh controller for restore, and deep-copies
mutable data. It supplies no current native entities, health or inventory.

`RestoreObservationMemoryBeforeDecision` maps the last-observed prefix to
native frame0 when the root is frame1. `combatObservation` then consumes the
actual root normally. No special case was added to valid action identities or
ordinary observation-memory restore.

`TestCombatCheckpointPreDecisionContinuation` passed using actual client
`combatObservation` calls on synthetic snapshots: active visible grenade,
GunFrame11, airborne/grounded states, occlusion, history, previous weapon
request, target intent, and a second nested save/load. Across32future decisions,
all871features, action proposals and exact sampling receipts match the source.
Corrupt prefixes and mismatched execution settings are rejected. This is a
controller-memory test, not native active-combat continuation evidence.

Full `go test ./...` passed. The first expanded test run rejected an intentionally
retained target after the test manually advanced decisions without the normal
`combatCommand` target cleanup; the fixture now performs that cleanup between
decisions and compares the second restored controller directly to the source.

Seeded native UDP regression passed again without source mutations during the
run: `workspace/artifacts/checkpoint-branches-20261116-210257-057/report.json`,
both `rng_restored=true`, two independent processes, source immutability and
mutation isolation accepted. It remains the quiet legacy scenario and does not
call the new combat-memory API. U1 hash rechecked unchanged.

Still required: wire capture BEFORE `combatObservation` and gate source command
dispatch on an explicit post-save native release acknowledgement; add a distinct
combat restore mode, suppress fixture reinitialization and rebind lockstep.
Planner navigation memory/context is separate from `combatControl`; its fidelity
must also be addressed before asserting identical871feature native continuation.
Legacy planner capture/restore restrictions remain intact. No fragments have
been collected or used for training yet.

## Native pre-decision source capture, 21:17 continuation

Connected the versioned combat path to client capture and the actual run loop.
The request specifies `boundary=combat_predecision_v1`; its sidecar contains
`combat`, never an idle `planner`/`runner` payload. Generic resume explicitly
rejects this type. The source client captures before `combatObservation`, holds
without consuming the root or sending its usercmd, and waits for an exact
ID/map/generation/frame release receipt. The checkpoint controller publishes
that receipt only after successful save and confirmation that the native
barrier cvar is zero. Failure releases the server cleanup barrier but does not
issue a success receipt to the client. Stale receipts cannot resume a new root.

Tests cover repeated hold callbacks, no history/action advance while held,
stale receipt rejection, matching-ID/wrong-anchor rejection, exact release,
no repeated capture, already-sent root rejection and legacy/combat payload
separation. Full `go test ./...` passed.

The first native source run reproduced an ordering defect in Yamagi
`src/server/sv_main.c`: `SV_TestLockstepHold` returned before the native
checkpoint acknowledgement. World/frame120 was held but the save was rejected.
Evidence retained in `workspace/artifacts/fragment-src-20261116-211116`.
Moved the isolated checkpoint hold/ack ahead of lockstep hold and clamped its
realtime as well. Built `q2ded` with the existing CMake/Ninja configuration
(`mingw64/bin` on PATH). Other existing native work was preserved.

The second run saved correctly and proved source continuation, but its outer
historical retreat/Shotgun teacher assertion rejected this learned-policy loss.
Retained `fragment-src-20261116-211325`; no quality result was changed.
The runner now has an explicitly scoped source-checkpoint admission path.
It checks the exact native applied command and its begin/end world-step receipts.

Fresh fenced source run accepted:
`workspace/artifacts/fragment-src-20261116-211551/source/report.json`.
Seed1729, mapbase1, native world120, history prefix119. Root command was
provider-owned, applied exactly once with every usercmd field equal to the
client trace, and advanced the native world once to121. Native save verified;
source fingerprint unchanged across the run (978Go/PS/module records).
The actor had a Shotgun at the root, GunFrame18. This root had no visible
projectiles; this is not an active-projectile or primed-grenade restore proof.

The rebuilt server also passed the complete seeded quiet two-process regression:
`workspace/artifacts/checkpoint-branches-20261116-211635-207/report.json`.
Both branches restored RNG and passed provenance/package/isolation checks.
An initial concurrent runtime-preparation attempt hit a Windows file-sharing
error; the sequential retry passed. No test processes remain running.

Durable machine-readable evidence:
`workspace/artifacts/hjepa-fragment-checkpoint-v1-20261116/native-source-capture-proof.json`.
U1 hash unchanged. No training, promotion, commit or push.

Next: dedicated combat restore and exact native continuation. Two specific
code findings need treatment: `g_test_combat_ready/released` are reset during
map spawn and are not serialized by the current savegame implementation;
stock `ClientBegin` changes `delta_angles` on load. Restore must neither rerun
fixture initialization nor refreeze combat/reset its RNG. H-JEPA's current
871features use v6 plus equipped/request context, not navigationFeatures;
planner navigation matters if control falls back and should not be claimed
restored by the combat-memory payload.

## Native combat load continuation, 21:42 continuation

Implemented explicit `checkpoint_mode=combat`. It accepts only coordinated
combat sidecars, sampled H-JEPA in the isolated synchronous harness, and the
first living native snapshot at frame1. It restores past memory before policy
observation, suppresses fixture placement/give/readiness, and holds the first
command until the controller verifies all combat restore receipts and clears
the native load barrier. Ordinary `resume`/`fresh` behavior remains separate.

Yamagi's opt-in `g_test_checkpoint_state=1` save trailer is now version2. It
stores combat readiness/released phase, allocation high-water/inactive-edict
freetimes, projectile receipt arrays and shot/enemy-bolt/Shotgun counters.
The normal RNG trailer still restores its exact cursor. The new state is
required on combat loads; missing/incompatible trailers are rejected rather
than inferred. Bound fragment clients preserve native `delta_angles` at begin.
Normal human-load behavior remains under the existing branch.

Two experimentally verified defects were fixed:

- Holding at networkframe1 initially allowed one game tick from frame0 after
  `ReadLevel`. GunFrame18 became19 and enemies advanced. The opt-in load now
  establishes networkframe1/time100 without running the restored world.
  The controller binds the `world_advanced=0` receipt to its new generation.
- `g_test_sg_serial` was static and absent from the first trailer. Actions and
  native commands matched, but native Shotgun trigger2 became trigger1.
  Trailer v2 retains this counter as well; the failed proof is preserved.

Retained implementation/debug runs:
`fragment-load-20261116-212609` (branch0 instance config override; branch1
fixture-placement validation; branch2 collected but had the extra world tick),
and `fragment-load-20261116-212945` (commands/features matched after the clock
fix, but full effect equality failed on the Shotgun trigger counter).

Final fenced native run:
`workspace/artifacts/fragment-load-20261116-213618`.
One source world, seed1729/base1, source root120, Shotgun GunFrame18. Two
independent new-process branches load the same package at nativeframe1.
Each branch passed `cmd/q2hjepa-fragment-proof` for30decisions (3game seconds):
26130feature values (871perdecision), exact proposals/sampling receipts,
interventions and usercmds, one native applied command/step per decision in
both source and branch, and17exact matching native gameplay effect records.
This includes native Shotgun pellet/trigger records and monster damage.
These are two mechanical repeats of ONE source world, not independent training
worlds or a combat-quality sample.

Proofs: `continuation-proof-{0,1}.json`; strengthened offline lineage/required
command-field checks also pass in `continuation-proof-strict-{0,1}.json`.
The strengthened audit does not mutate any captured inputs. Both Go/PS and
native source fingerprints were unchanged throughout native collection.
`source-manifest.json` binds source/native files and client/server/game hashes.

The checker refuses missing/duplicate native application/steps, checks saved
package/model/root lineage, compares actual Go features and sampled proposals,
and aligns native effects by the command that caused them, including the last
world step while excluding the source's next command. Logged navigation may
differ after planner route reconstruction; current H-JEPA871inputs do not
consume navigationFeatures. This is not a restored planner-fallback guarantee.

All Go tests and native CMake game/q2ded builds passed. Unit tests cover the
first restored observation after frame0 memory, held release, setup suppression,
native command duplication/missing step and effect-window attribution.

An additional old quiet-suite run `checkpoint-branches-20261116-213926-454`
exposed a Windows file-sharing race: `CheckpointActor` exited while reading
the just-published control file, so participant collection timed out. The
client now waits/retries Win32sharing errors32/33 before dispatch, and a held
combat root likewise continues waiting if its release file is briefly locked.
The failed run remains retained; this is separate from the successful native
combat continuation evidence.

After the sharing-error fix, the full seeded quiet two-process suite passed:
`workspace/artifacts/checkpoint-branches-20261116-214145-104/report.json`.
Both load receipts restore RNG, package copies and provenance match, source
files remain immutable and mutations stay isolated. A Windows-specific test
then reproduced actual exclusive file locks with `CreateFile(share=0)` and
verified that request/release locks preserve the pending combat boundary and
become usable after unlock. The final `go test ./...` and `git diff --check`
passed. No owned workers remain live.

Aggregate evidence:
`workspace/artifacts/hjepa-fragment-checkpoint-v1-20261116/native-combat-continuation-proof.json`.

Still open: active projectile/primed-grenade save/load proof; distinct sampled
branch streams and typed fragment export; truncated-return/long-tail handling;
actual training and independent full-battle validation. No fragment training
or promotion has taken place. U1 remains unchanged.

## Grenades and frozen-model sampling streams, 22:04 continuation

Diagnostic fixture and evidence root:
`workspace/artifacts/fg-20261116-214800`.
It uses base1/seed1729, stock vulnerable Parasite,20Grenades plus fallback
Blaster, and exact native inventory startup. Frozen actor parameters come from
the existing sampled actor bridge. The diagnostic model explicitly keeps the
equipped grenade; these captures are fidelity probes and are excluded from
training/quality evidence. A second model changes categorical fire/pose
temperature from0.25to2, with unchanged actor parameters. The first attempted
warm JSON used the wrong `temperature` key and was rejected; the failed
`flight-source` and original JSON are retained. `keep-grenades-warm-v2.json`
uses the actual `categorical_temperature` contract.

Primed grenade continuation passed: source root125, GunFrame11, no released
projectile yet. Its native in-hand explosion at serverframe142 (gameframe144,
mod24, HP100to41) is reproduced exactly. The latest engine repeat is
`prime-source-v4`, `prime-branch-v4`, `prime-proof-v4.json`:30decisions,
26130matching features, exact action/sample/native-step receipts and the
matching explosion/damage record.

Flying grenade initially failed: source root116 has observed `hand_grenade`
entity67 already in flight. Loaded player recoil changed from[1,0,-1.5]to
[0,0,-0.25], causing feature811tochange. The normal `ClientBegin` called
`ClientEndServerFrame` a second time and consumed saved view/damage feedback.
The opt-in bound-fragment begin now preserves the complete saved playerstate
and publishes only its native inventory packet. Simply skipping end-frame
first exposed the transport's missing inventory cache; this failed engine-v3
run is retained. Engine-v4 rebuilds that transport cache with `InventoryMessage`
without modifying playerstate. Ordinary nonfragment begin remains unchanged.

Flying continuation now passes on the latest engine:
`flight-source-116-v4`, `flight-branch-116-v4`, `flight-proof-v4.json`.
All30decisions/871features, samples, commands and6native effect records match.
Both current-engine primed/flying proofs pass; CMake game build succeeded.
`engine-v4/engine.json` and `native-sources.json` pin its actual binaries/source.

Added explicit fragment sampling streams while retaining the same frozen
model bytes/version. `hjepa_sampling_checkpoint_v2` carries a nonzero stream;
v1/default stream0 preserves all original draws. A domain-separated SHA256
derives primitive and weapon draw seeds from each frozen base seed and stream.
The density/distribution is unchanged. `Sample.sampling_seed` stays the model's
base seed; exact replay additionally requires the logical checkpoint identity
and stream. Nested captures retain both. Runtime configuration is explicitly
`test.checkpoint_sampling_stream` and is allowed only for combat restore.
The restore receipt records it; exact-continuation proof refuses mismatched
streams rather than confusing a fork with a replay.

Policy tests cover unchanged model identity/density, different primitive AND
weapon draws, same-stream replay across native sessions, nested continuation,
and invalid stream rejection. Native branches under `forks/` use streams0,1,2,
and1again. Stream0 passes source continuation equality. Different streams1/2
share identical871root features and produce27different comparable proposals.
Stream1 has27sampled decisions before `system2_noncombat` control transfer;
stream2 has30before the horizon. The verifier now records that boundary and
never labels the subsequent rules tail as neural samples. The full30native
steps remain verified. Repeat stream1 reproduces all30native steps/features,
its27sampled decisions and native effects exactly.

`forks/different-stream-proof-v2.json` and `repeat-stream-proof.json` bind model,
source package, streams, raw captures and actual Go actor/density/draw replay.
The original fixed30-neural-row proof failed at transfer step27 and is retained
as `different-stream-proof.json`. `forks/source-manifest.json` binds the frozen
Go/PS collection sources; fingerprints were unchanged during collection.
Final `go test ./...` and `git diff --check` passed. U1 hash unchanged.

Next: typed fragment transition exports, explicit terminal/control-transfer/
horizon handling and delayed-effect targets, then training on genuine TRAIN
banks with the full learned weapon head and fresh full-battle validation.
The current keep-equipped diagnostic probes must not become training labels.


## Typed TRAIN fragments and first joint CUDA update

The distinct `hjepa_native_fragment_v2` export carries actual and logical
sampling identities, frozen model/package/plan hashes, 871 observations and
next observations, actor logits, primitive/weapon log probabilities, proposals,
sent commands, interventions and causally joined native damage receipts.
A horizon or a transfer to another controller remains a truncation requiring
bootstrap; it cannot silently become a terminal with zero tail value. Raw
fragments are never optimizer-ready. The existing full-episode loader is not
used for these fragments. Diagnostic keep-equipped grenade captures remain
excluded from training.

`q2hjepa-fragment-proof --export-fragment` verifies a registered TRAIN condition
plan and its exact generated fixture plus native start receipt. The plan
specifies the condition; the frozen full learned H-JEPA actor is bound separately
by capture/model hashes. Its primitive and weapon decisions are independently
replayed in Go. A separate `--credit-prefix/--credit-tail/--credit-out` operation
requires a longer continuation of the SAME source root, behavior model and draw
stream. It compares the entire short prefix, allowing only process-local
receipt identities and horizon markers to differ. Native death or closure of
the nonspawning initial monster roster establishes the tail value; unresolved
tails are rejected. Pre-root monster deaths are selected by causal native step
end, not damage-event frame alone. Rewards are
`100goal-100death+10capped_progress-.1received-self-.5team_v1`.

Pilot root: `workspace/artifacts/fragment-train-20261116-221723`.
The sole TRAIN condition is registry offset 511, base1 soldier site 017 random
roster. Offset 520 was rejected before collection because TRAIN has only 512
registered offsets. `source-0` captures before decision 110. Independent native
branches `branch-0-1` and `branch-0-2` supply 30 sampled decisions each. Their
longer same-stream continuations `tail-0-1` and `tail-0-2` reproduce those short
prefixes exactly and clear the initial roster after 82 and 66 neural decisions.
The authoritative exports are `v2-prefix-{1,2}.json`, `v2-tail-{1,2}.json` and
`v2-targets-{1,2}.json`; older prototype v1 files are retained but not consumed.
Root returns are 101.75454545454544 and 102.65454545454544. This is ONE checkpoint
family with two stochastic streams, not 60 independent worlds.

`scripts/train_combat_hjepa_fragments.py` performs bounded CUDA float64 joint
primitive/weapon PPO with an independent-stream return baseline, masked weapon
categorical probabilities, clipped ratios, gradient clipping and mean-KL-limited
backtracking. It updates the first encoder, primitive actor/log std and separate
weapon actor; auxiliary world heads are frozen. TRAIN source hashes, unique
family/stream keys and exact initial Go/CUDA logits/density are admission gates.
The first update used RTX 5070, 60 decisions, mean KL 0.006300208329442129 and
step scale 0.015625. `fit-1/model.json` SHA256:
`5a0edb132a2816935fd0e5244c0c973e793062da3d4df473f9ef739889e959bf`.
Go/CUDA actor and weapon parity both passed (maximum absolute error
2.6645352591003757e-15); initial joint density error was 1.0125233984581428e-13.
Direct two-update training and one-update-plus-resume produced byte-identical
models (`resume-check.json`). Fit-1 was fixed as the candidate before validation;
fit-2 only checks deterministic resume. No live model was replaced.

Fresh validation offset 448 failed during condition generation before gameplay
(city1 soldier-ss actor overlap after 128 attempts); it is not a model loss.
Offset 449 generated 16 worlds across base1/base2/base3/bunk1/ware1/jail1/hangar1/
city1. Parent and fit-1 use deterministic mean primitives/greedy weapon in this
screen. Their 32 native captures are valid: parent 1 win / 10 deaths / 5 timeouts,
1059 received HP and 73 self HP; candidate 2 wins / 9 deaths / 5 timeouts,
1026 received HP and 72 self HP. This small screen is not promotion evidence.
The first U1 comparison launcher erroneously specified legacy_v1; U1 declares
stock_weapons_fire_v1 and correctly rejected that mismatch before gameplay.
Failed reports remain under `screen-449/native/u1`. Corrected U1 and rules
comparisons use fresh `screen-449/baselines-corrected` output.


Corrected baselines completed with all 32 captures valid: U1 8 wins / 4 deaths /
4 timeouts, 718 received HP and 65 self HP; rules 8 wins / 3 deaths / 5 timeouts,
567 received HP and zero self HP. `screen-449/native-screen.json` joins all 64
valid games and paired results. Candidate gains one ware1 win (case 9) and
avoids one death without winning in case 2; all other outcome classifications
match the parent. These are descriptive results from one small candidate screen,
not evidence that fragment training beats U1 or rules. No promotion. U1 bytes
remain unchanged (SHA256 21c3b680e0b6090d601d524f6da53d73ae6dd083c8e7a9a6da2e6d268ccadd26).

Final validation: full `go test ./...`, `git diff --check`, nine Python fragment
admission tests, successful loading of the actual 60-row TRAIN pair, actual Go
actor/weapon parity and exact direct/resumed CUDA fit passed. Loader tests cover
source mutation, missing prefix hashes, duplicate mechanical streams, a missing
paired stream, diagnostic admission, nonfinite returns, model mismatch and
horizon-as-terminal rejection. Required prefix/tail/model/plan hashes and all
fragment source hashes are checked before optimization. All collectors are
terminal. Preserve all prototypes and failed reports.

Next experiment: collect multiple independent TRAIN checkpoint families across
maps, monster classes, initial weapons and combat phases; keep paired sampling
streams within each family. Lock a new candidate before a new validation cohort;
449 has now been viewed and must not become TRAIN data or an untouched holdout.
Keep terminal MC targets until a separately tested value bootstrap can handle
unresolved tails. This pilot demonstrated end-to-end mechanics, not sustained
combat quality or reduced total training cost. Long continuations are still
needed here for terminal credit even though only short prefixes enter updates.


## Expanded independent checkpoint bank

`workspace/artifacts/fragment-bank-20261116-224759` freezes a registered TRAIN
plan at offset 510: eight source worlds on base1/base2/base3/bunk1/ware1/jail1/
hangar1/city1. All eight source captures before decision 110 passed, with the
full sampled learned primitive and weapon actor controlling the root. Initial
weapons include HyperBlaster, Blaster, Railgun, Machinegun, Grenades and Shotgun;
rosters include soldiers, infantry, berserk, gunner, medic and flyer. Sixteen
independent native 100-decision continuations passed typed v2 export/replay.
`protocol.json` and `source-jobs.json`/`branch-jobs.json` bind their unchanged
collection sources. There were no source or branch job failures.

Most 100-decision tails were unresolved, demonstrating why a horizon cannot be
assigned zero bootstrap. Ten longer continuations replayed the SAME saved
root/draw streams with a 500-decision budget. `long-protocol.json` separately
binds the changed collection helper; client/engine/model bytes remain the same.
The fragment/export CLI and branch helper now allow up to 500 decisions for
credit tails. Fitting windows remain 30 decisions. Each admitted longer tail
exactly reproduces its original 30-decision prefix before supplying credit.

Native actor death is a valid terminal even when an initial roster includes a
medic or unsupported monster lifecycle. Such a roster cannot prove a goal just
by exhausting its initial entity IDs: resurrection/spawning breaks that proof.
`nativeTerminalRewards` now gates roster-clear goals separately from actor death;
unit tests cover delayed kills, overkill caps, unresolved horizons, an unsafe
medic roster, simultaneous actor death/goal (death takes precedence), missing
next dead observation, and nonmutation of the caller's source roster.

Five new checkpoint families have paired native-death targets: base1, bunk1,
ware1, jail1 and hangar1. Their terminals occur at 70..306 decisions. Base2,
base3 and city1 are excluded from this fit because a controller transfer occurs
before a proven native terminal; their captures and rejection receipts remain.
This is a selection limitation, not representative coverage of all eight maps.
Adding the prior base1 offset511 success family gives SIX independent families,
12 streams and 360 short decisions across FIVE maps. New families are losses;
the prior family supplies the two success streams. `admission.json` records all
accepted/excluded families and exact target paths. Unresolved tails have not
been relabeled as losses or assigned synthetic returns.

A single fixed joint CUDA update uses the original sampled behavior model and
this 360-row bank, not the prior fitted candidate as behavior. `fit-1/model.json`
SHA256 `0175abdd5db07873aa0c174c25d1d85dde8cc28d0ee682509ad08a9f4e6735f3`.
Mean KL is 0.005667785552525907, maximum KL 0.026013513388215285, backtracking
scale 0.015625. The constraint is explicitly on MEAN KL. Initial Go/CUDA density
error is 1.1191048088221578e-13. Actual Go actor parity passed at
3.552713678800501e-15; weapon parity passed at 2.6645352591003757e-15.
No model promotion or live U1 replacement.

Fresh validation450 was generated successfully. `screen-450/protocol.json`
fixes 16 worlds across eight maps and five variants before gameplay: original
H-JEPA mean actor, one-family pilot mean actor, expanded-bank mean actor, U1
(stock weapon execution contract), and legacy rules. All H-JEPA variants use
mean primitives/greedy weapon so stochastic collection noise is excluded from
this specific evaluation. Full terminal Monte Carlo collection is still needed
for this pilot's short-prefix targets: no reduced total collection-cost claim.


The three H-JEPA arms of validation450 completed with valid native captures:
original 3 wins / 8 deaths / 5 timeouts; first one-family pilot 4 wins / 6 deaths /
6 timeouts; expanded bank 4 wins / 8 deaths / 4 timeouts. Received HP totals are
1037 / 1048 / 1060 respectively. Thus simply expanding this terminal-MC bank did
not produce a clear quality improvement. On base1 case0, the bank candidate
turns a parent/pilot win into death. It gains wins on base2 case3 (also gained by
the first pilot) and base3 case5. It turns timeouts into deaths on ware1 case9
and jail1 case10, while avoiding a death without a win on jail1 case11. These
observed cases should guide a diagnostic component split between the primitive
encoder/actor changes and weapon actor changes before collecting more of the
same data. Once viewed, validation450 is a diagnostic cohort, not untouched
validation for another tuned candidate. It must never become TRAIN.


All 80 validation450 games completed and passed native capture/provenance checks.
U1: 4 wins / 8 deaths / 4 timeouts, 1071 received HP, 89 self HP. Rules:
5 wins / 6 deaths / 5 timeouts, 824 received HP, 73 self HP. The bank candidate
matches U1's outcome counts in this one cohort, but does not establish equal
strength or superiority; the pilot has fewer deaths and rules have more wins.
Bank self HP is 74 versus 46 for both parent and pilot. Different baseline counts
between cohorts449 and450 illustrate why their raw wins must not be treated as
a before/after effect. `screen-450/native-screen.json` stores all 80 valid reports,
paired seeds/maps and outcome changes; `summary.json` records the closed pilot.

Final tests: full `go test ./...`, ten Python admission tests, strict loading of
the actual 360-row bank and `git diff --check` passed. Added a 306-row typed-tail
boundary test and an optimizer rejection test for prefixes over 100 decisions:
long credit continuations cannot silently become long fitting windows. Native
terminal-credit tests and actual 500-budget native collections passed. U1 bytes
remain unchanged. All source, branch, long-tail and matrix jobs are terminal.
No commit or promotion was performed by this run. Goal of autonomous learned
combat stronger than rules remains open. Next is a diagnostic component split
on already-viewed450, followed by genuinely fresh validation for any new fixed
candidate; increasing this same selected bank alone is not the next evidence-
based action.


## Component diagnosis on viewed validation450

48 additional games under `fragment-bank-20261116-224759/component-450` separate
primitive encoder/actor versus the independent weapon actor, plus 16 parent
repeats. All captures are valid. Weapon-only: 3 wins / 8 deaths / 5 timeouts,
1037 received HP and 46 self HP, exactly the parent's counts. Primitive-only:
4 wins / 8 deaths / 4 timeouts, 1060 received HP and 74 self HP, exactly the joint
bank candidate. Stronger than matching totals, all 16 first-life command and
proposal traces are exactly equal for weapon-only versus parent and primitive-
only versus joint. `ablation-equality.json` records this conditional finding:
weapon probabilities changed, but their greedy choices do not change these
16 trajectories. This is not a claim about sampled actions or other states.

The parent repeat reproduced every first-life command/proposal, native damage,
observed last states and outcome/HP counters. Initial whole-file equality was
too broad: asynchronous launcher stopping leaves variable extra postterminal
records, entity arrays in summaries are unordered, and two auxiliary aim/source
counters differ by one frame even with identical actual commands. These raw
captures and the failed unscoped `parent-repeat-proof.json` are preserved.
`parent-semantic-repeat-proof.json` explicitly scopes successful comparison to
first-life gameplay, sorts only visible-enemy/grenade entity arrays by ID, and
records auxiliary counter differences. It does not claim byte-identical reports.

Another 32 valid diagnostic games under `primitive-parts-450` split the two
primitive components: encoder-only has 3 wins / 7 deaths / 6 timeouts, 1003
received HP, 46 self HP; actor-only has 2 wins / 9 deaths / 5 timeouts, 1051
received HP, 46 self HP. The encoder-only swap reproduces base1case0's lost win;
actor-only preserves that individual win but performs worse overall. These
counterfactual swaps were not themselves trained, so their results do not
justify declaring a frozen-encoder training method superior. Keep both as
negative/diagnostic evidence rather than selecting a new promoted policy.
`component-450/summary.json` binds the component findings.

Previous fresh449/450 screens evaluated mean primitives and greedy weapons,
whereas fragment PPO fits a sampled joint distribution. To avoid mistaking
that deployment-policy difference for a learning failure/success, `sampled-451`
fixes fresh validation451 with the untouched sampled parent, first pilot and
expanded bank models, plus U1 and rules. No new weights are selected/fitted.
Only one paired sampled trajectory is run per world: this remains a small
screen, not a precise expected-return or superiority estimate.


## Correction: native objective requirement was dropped by matrix launcher

The completed sampled451 screen has legacy initial-roster goal counts of
parent 5 / pilot 6 / bank 6 / U1 7 / rules 4, and deaths 6 / 6 / 6 / 5 / 5.
These are preserved as DIAGNOSTIC legacy-goal counts, not fully objective-
verified combat wins. Three independent Go frozen-model replays passed with
zero proposal mismatches: parent2396, pilot2422, bank2384 provider frames.
`sampled-451/summary.json` records the limitation. No candidate was promoted.

Code audit found a real launcher bug: these registered recipes specify
`require_native_objective=true`, but `run_combat_hjepa_matrix.ps1` did not carry
it into `run_learned_combat_baseline.ps1`. Captures used goal_v1 (verified native
kills of initial classes), while recipe fidelity requires goal_v2 with native
alive=0 and pending=0. Initial kills alone cannot prove completion for medic
revival or boss phases. Earlier449/450/451 results remain useful for conditional
command/component diagnostics, but must not be cited as fully objective-verified
victories or promotion evidence. Existing TRAIN terminal-MC fragments use a
separate conservative native death/nonspawning-roster closure proof and are not
invalidated by this full-episode launcher bug.

The launcher now preflights required engine capability before creating output,
passes RequireNativeObjective per task, records it in each job, and rejects a
result missing the flag or a goal receipt not v2. The original engine-v4 manifest
has no objective capability and is now correctly rejected before any games.
Actual pinned game/server binaries already contain the existing objective
observer. `engine-objective-v1` is a NEW manifest/copy of the SAME binaries/source,
not a recompiled or modified game, and declares the capability only with native
proof under `objective-check-451`. Both native probe games passed: an ordinary
base1 win has goal_v2 at frame144 with alive0/pending0; the medic-containing case
remains supervised without a fabricated goal. `objective-check-451/proof.json`
records positive native events and negative preflight. The observer is
supervisor-only and does not become a policy feature.

Fresh `objective-sampled-452` fixes the same original/pilot/bank sampled model
bytes plus U1/rules before collecting 80 paired games on 16 new worlds/8 maps.
All cases now require the recipe's native objective, and any accepted victory
must carry native alive0/pending0 evidence. Sampled451 is already viewed and is
not reused as untouched quality evidence. No new optimizer updates were made
in this component/measurement correction phase.


The corrected sampled452 screen completed all 80 games with valid captures,
required objective flags and independently rejoined native goal_v2 records:
parent 2 wins / 10 deaths / 4 timeouts (1155 received HP, 257 self HP);
one-family pilot 4 / 10 / 2 (1217 received, 226 self);
expanded bank 4 / 10 / 2 (1217 received, 232 self);
U1 7 / 8 / 1 (857 received, 124 self);
rules 9 / 3 / 4 (556 received, 45 self).
`objective-sampled-452/summary.json` stores all 80 report paths, exact native
alive0/pending0 goal receipts, per-case control ownership and model verification.
All actual H-JEPA model JSON values match the fixed files. The baseline harness
rebinds U1's sampling_seed to each episode seed; all other JSON values match the
frozen source, whose bytes remain unchanged. No promotion or superiority claim.
Both fitted candidates gain completed goals versus the parent on this small
cohort, but neither reduces the ten deaths; both remain well below U1/rules.

The sample screen still includes rules ownership during noncombat/visibility
fallback: parent114, pilot245, bank243 living first-life postrelease frames.
Do not attribute every outcome to uninterrupted fully learned control. The
broader user goal of full autonomous learned combat remains open.

Two-case objective observer noninterference proof shows exact first-life commands
and proposals with/without telemetry (ordinary goal through144, medic case
through398): `objective-check-451/observer-noninterference.json`. This extends
capability proof beyond just a readiness banner. Existing Go Goal/Objective
verification tests passed. The full native corrected matrix provides the live
regression check for requirement propagation, including U1 and legacy rules.

Focused diagnostic `objective-sampled-452/failure-diagnosis.json` records cases3,
4 and13. In base2 case3 both successful U1/rules and failed H-JEPA use Blaster;
H-JEPA deals40 monster HP versus140, takes100 enemy HP versus34/18, and has zero
self damage. Thus self damage or weapon selection alone cannot explain that
failure. In base3 case4 all three have45 self HP, while enemy damage differs
55/9/3. In hangar1 case13 the candidate deals304 machinegun HP plus300 rail HP
without a kill and dies, while U1 clears850 HP with rail and takes no damage;
this motivates examining aim/fire/movement and weapon timing, not a proven
single cause. New TRAIN seeds with these condition types, not the viewed452
states, are the next collection targets. Native-objective-backed fragment goal
credit can broaden the conservative initial-roster closure rule for advanced
monsters; unresolved controller transfers still need explicit handling.

Final checks after the objective propagation fix: full `go test ./...`, focused
Go Goal/Objective tests, negative preflight, two native capability/noninterference
probes, all80 objective-certified games and `git diff --check` passed. No new
optimizer update or model promotion in this phase. All observed process handles
are terminal. Other staged/user changes are preserved; no commit/push performed.


## Native-objective fragment credit and balanced bank, November17

Artifacts: `workspace/artifacts/fragment-objective-20261116-233944`.
`objective_credit.go` joins independently parsed supervisor counters to every
next observation using map, generation, network frame and preserved world game
frame. The source root must have alive/pending work. Restored tail game frames
must equal the source game frame plus the transition index plus one. Enabled
but missing, stale, differently clocked or malformed objective telemetry cannot
fall back to initial-roster closure. Native actor death keeps precedence over a
simultaneous clear; a goal needs a living next observation. Legacy no-telemetry
captures retain their separate conservative proof. New target reason
`native_objective_clear` includes the exact terminal counter record. The CUDA
loader checks zero alive/pending and its terminal-next-identity binding.

A new capture/streams1/2 of known TRAIN511 reproduces root returns
101.75454545454544 and102.65454545454544, now through native objective closure.
The old native-roster target still reproduces its old value. These are mechanic
verification captures, not an additional independent world. Re-recording the
same fixture/root creates a different package family ID; the loader now hashes
the source-bound generated TRAIN condition, logical root frame and behavior
model to reject that duplicate. It also reports unique training_worlds separately
from checkpoint families. Actual old+new known511 recapture admission rejects;
the original360row bank still loads correctly.

Initial-condition-only selection from a registered TRAIN480..511 plan uses two
base2 Blaster worlds (seeds1823180482/483) and two hangar1 tank worlds with Railgun
available (1904200485/487). Only one of32 tank candidates initially equips Railgun;
the second deliberately starts Blaster while owning Railgun, exposing choice.
The base2 initial selections include flyers; they are exploratory conditions,
not an exact match of the earlier ground-only loss. A later pre-game roster
filter chooses ground SS+infantry TRAIN seeds1823180484/491, with two/three
monsters respectively. Earlier conditions/captures are preserved.

All six new source captures and48 independent streams1..8 restore correctly.
Native terminal credit, not the last raw capture boundary, determines outcome:
initial family0 has1goal/7deaths;family1 has4deaths/4unresolved; tank family2 has
8deaths; tank+medic family3 has6deaths/2unresolved. Ground family0 has8goals;
ground family1 has7deaths/1unresolved. Total9native goals,32deaths,7unresolved.
One initial family0 stream6 clears at transition44 (network46, game157), despite
its later visibility-transfer boundary. A preliminary last-boundary-only count
missed that victory and was corrected after native-counter/credit verification.
This is why sampler cutoff types must not substitute for task outcomes.
All first30decision prefixes end before native terminals. Unresolved prefixes
remain bootstrap-required and are excluded from this fit, not zero-labeled;
this creates an acknowledged admission selection limitation.

Combine41resolved new streams with12old streams:1590decisions,53streams,
12checkpoint families and12unique TRAIN worlds across six maps. Known511
verification is deliberately absent. `admission.json` binds every exact target.
`--family-balanced` gives each family equal surrogate and mean-KL mass despite
unequal stream counts. Its explicit optimizer-algorithm key rejects incompatible
resume; the default row-weighted path remains byte-compatible. A gradient
invariance test confirms that duplicating rows in one family does not increase
its influence under equal-family weighting. Thirteen Python admission/weighting
tests pass, including actual recapture rejection and native objective binding.

Four fixed CUDA updates on RTX5070 produce
`fit-balanced-4/model.json` SHA256
`557056930e0bcc30dc28fe8513d9ff35f06087a9b7588c4bff2f1fa5a6a72718`.
Final family-weighted mean KL0.006144553018883045, max KL0.11895114471890451;
the gate is on the weighted MEAN, not a per-state maximum. Initial joint-density
parity error1.7363888105137448e-13. Actual Go actor/weapon parity errors are
2.6645352591003757e-15 and1.7763568394002505e-15. Direct4 and2+resume2 produce
identical model bytes. Default legacy old1+new1 reproduces the historical2update
model exactly. Omitting the balanced flag while resuming a balanced checkpoint
is rejected without an output directory. Sources/runtime versions are pinned
under `training-source`; the fixed step4 candidate predates validation.

Fresh objective-required sampled453:80 valid games/16pairedworlds/eight maps.
Parent4wins/8deaths/4timeouts,1182receivedHP161selfHP;
previous360row bank3/7/6,990received137self;
newbalanced1590row candidate2/7/7,997received98self;
U1 6/6/4,820received170self;
rules8/2/6,466received79self.
Every goal_v2 is independently joined to native alive0/pending0; actual model
JSON values match frozen files (U1 only rebinds its episode sampling seed).
No improvement or promotion. Lower self HP is not enough to establish better
combat. `screen-453/summary.json` includes per-case controller ownership and
raw-report paths. A secondary audit finds native empty-world states without the
legacy actor-kill receipt in ware1case9 for U1/rules; these are recorded separately,
not silently added to the predefined primary score. They do not change the
negative candidate conclusion. Viewed453 must not become TRAIN/untouched data.

The TRAIN-only `fire-exploration-audit.json` identifies a concrete coverage gap:
ZERO hand-grenade observations in all1590fitting rows. Credited tails contain
7030provider rows before native terminals,45hand-grenade rows,33primed gunframe11
rows and ZERO sampled releases there. Frozen logits imply only
0.0000022431448494314817 expected releases at current temperature0.25; a read-only
fire-temperature1 calculation gives0.4540061123102805 expected releases.
This is not a native warm-policy evaluation or proof of improvement. Earlier
natural-actor notes also documented poor release exploration; current data
confirm that the early fixed110/first30 windows omit the relevant state entirely.
Next: event/phase-targeted mid-battle checkpoints and declared fire exploration
while preserving pose sampling and exact Go/CUDA likelihoods. Do not add a fixed
release controller or infer global-temperature effects are fire-only. Continue
full learned control toward U1/rules quality rather than repeating the same early
prefix fit. Existing utility/world heads remain inactive under plannerbc; this
phase trained joint actor/weapon control, not a paper H-JEPA world-model replica.

Final validation: full `go test ./...`,13Python tests,diffcheck,actual native
objective/legacy proofs,Go/CUDA parity,exact balanced+legacy resume and mismatch
rejection passed. All native/training/postprocessing handles are terminal.
U1 remains SHA25621c3b680e0b6090d601d524f6da53d73ae6dd083c8e7a9a6da2e6d268ccadd26.
Other staged changes remain intact; no commit/push/promotion was performed.

## Observed grenade-phase checkpoints and independent fire exploration, November17

`workspace/artifacts/fragment-phase-20261117-004551` contains the next experiment.
`CheckpointCapturePhase=hand_grenade_primed_v1` arms a minimum frame rather than
assuming a fixed frame is interesting. Before consuming the next observation or
choosing an action, the client waits for living first-life equipped Grenades at
observed gunframe11, holds that exact root, and resolves the native checkpoint
barrier to it. This reads public observations only. Gunframe11 alone does not
prove a private native fuse age; the first observed11 can precede initialization
of grenade_time in that tick. No release action or weapon choice is forced.

An optional `primitive_sampling.fire_temperature` controls Bernoulli fire only;
omission/null preserves the legacy categorical temperature for fire. Pose keeps
its original temperature, and Gaussian latents/seeds remain unchanged. Go draw
and likelihood replay, CUDA PPO/KL, natural/Fisher consumers and fragment fitting
use the same declared fire temperature. Independent distribution/Hessian tests
cover pose0.25/fire2 as well as the legacy path. The warm source changes only
fire_temperature=2 in the original sampled parent, SHA256
`cef0d0855f619bc1be7550be36bec418333cbac42d4b714593b038aa6b2e4b1c`.
It retains full learned primitive and masked weapon control, plannerbc, no motor
or held weapon option. Cold and warm behavior policies are not mixed in fitting.

A verified TRAIN384..447 plan supplies64 initial conditions on each of base1,
bunk1,jail1,hangar1. Eight conditions are selected before gameplay from initially
equipped Grenades with ammunition, preferring fewer other owned guns. Seven
encounter the phase and save at actual frames109,158,109,109,110,114,114; source5
does not encounter it and is retained as a failed acquisition, not admitted.
Each of the seven source observations exactly matches all871 loaded-root
features under `--root-only`; each first counterfactual action/density/native
command is independently replayed. A separate same-stream30decision native
continuation matches26130 feature values, actions and logged effects. The actual
warm Go/CUDA audit includes24fire-on and6fire-off samples, raw max error
1.7763568394002505e-15 and joint-density error6.217248937900877e-14.

All56 new native500-budget streams1..8 finish without technical capture errors.
Native terminal credit establishes11objective clears,38deaths,7unresolved.
The unresolved streams retain a required bootstrap and receive no zero label;
excluding them is an acknowledged selection limitation. Earliest terminals and
available decisions determine common within-family fitting windows of5,5,30,
30,30,30,8. Every exported prefix is replayed against the same-stream longer
tail; credit fails if it crosses the true terminal. A one-decision export is now
supported, although this bank needs at least5. The admitted bank has919rows,
49streams,7families and7unique TRAIN worlds on4maps. Unlike the previous bank,
790rows equip hand grenades,282are observed gunframe11, and51sample fire-off
there. `native-grenade-audit.json` verifies actual thrown-grenade damage in all49
credited tails: mod15/16 inflict5629monsterHP and1877selfHP before the exact
terminal, with no mod24 held-explosion damage. Damage-burst groups are not a
complete projectile-spawn count or proof of safe throwing.

Two candidates use the same bank, parent, seed1729, learning rate and fixed4CUDA
updates. Both train the full shared encoder0/primitive/weapon networks and log
standard deviations. Equal-family weighting is compared with
`--family-balanced --phase-balanced`, which additionally gives equal mass to
present hold11/other observation strata within each family. It never conditions
weights on the sampled action or return. A family with only one stratum retains
all its mass. This explicitly changes the state-weighted TRAIN objective, not an
unbiased estimate of the original episode objective. Resume binds this algorithm
key and rejects changing it.

`fit-family-4/model.json` SHA256
`a9acb0a60939f87b903e5e0f981aaa4f27026aad41e02de534aa91766884c11d`,
final weighted meanKL0.002942927005850224/maxKL0.04651632630809815.
`fit-phase-4/model.json` SHA256
`9bf0c868578f530a0d305418705ba85467dccde89cc8311d4c396eaa4c0e66c4`,
meanKL0.009535305031232373/maxKL0.07568459311447037. Only mean KL is gated.
Direct4 and2+resume2 phase fits produce identical model bytes. Both candidates
pass actual Go actor and weapon parity. The TRAIN mean release probability on
the282hold11states changes only from0.1975234780 to0.1973656359/family and
0.1973102707/phase; this is a state diagnostic, not combat improvement.
Source snapshots, hashes, targets, rejected-credit receipts, update histories
and all native evidence are retained under the experiment root.

Full `go test ./...` and29focused Python tests pass. Fresh objective-required
validation454 is preregistered with16worlds on8maps and six frozen variants:
cold parent,warm parent,equal-family fit,phase fit,U1,rules. The warm parent
comparison separates exploration from learning. Both fixed step4 candidates
predate viewing outcomes. Until the complete independent screen is assessed,
there is no quality or promotion claim; U1 remains unchanged.

Validation454 is now closed: all96captures are valid and objective-required;
the Go/PowerShell source fence is unchanged throughout the capture. Frozen
models match actual runtime JSON (U1 only rebinds sampling_seed to the episode).
Every primary goal receipt independently matches native alive0/pending0.

| Variant | Wins | Deaths | Timeouts | Received HP | Self HP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold parent | 4 | 8 | 4 | 1144 | 298 |
| Warm parent, fire2 only | 6 | 7 | 3 | 991 | 188 |
| Equal-family four updates | 3 | 9 | 4 | 1072 | 249 |
| Phase-balanced four updates | 3 | 8 | 5 | 1045 | 139 |
| U1 | 4 | 8 | 4 | 883 | 74 |
| Rules | 7 | 2 | 7 | 502 | 125 |

Rules ware1case9 additionally has a living native clearance without the legacy
actor-kill receipt, recorded as secondary only. All other secondary native
clearances agree with the primary score. Both fits retain warm-parent wins on
base2case3/bunk1cases6,7, but lose its wins on base3case5,jail1case11 and
hangar1case12. The warm-only6versus cold4 is a small validation observation,
not statistical superiority or grounds to replace U1. Neither fitted candidate
improves its own behavior parent; no promotion or best-step reselection.

`train-kl-component-audit.json` identifies an optimizer diagnostic: family-fit
meanKL0.0029429 is almost entirely yaw0.00193576 and pitch0.000723906 plus
movement; fire contributes only8.51194e-8. Phase-fit meanKL0.0095353 contains
yaw0.00346238,pitch0.00447289 and fire1.76099e-7. These are exact TRAIN
distribution-change components under each candidate's own weighting. They
support the narrow conclusion that adding held-grenade rows and state weights
hardly changed fire probabilities. They do not establish which changed
component caused the held-out losses. Prior full natural-gradient experiments
already failed a quality screen; do not relabel that same remedy as new.

Next investigate paired branches with common movement/pose/weapon random draws
and separately varied declared fire randomness, to reduce noise in the fire
credit estimate while retaining learned joint control. This is a proposed next
experiment, not an implemented sampling contract or a demonstrated improvement.
Also evaluate using unresolved tails through a verified learned bootstrap before
claiming fragment sample efficiency; current Monte Carlo labels still require
long continuations. Viewed454 must not become TRAIN or untouched confirmation.

The remaining actor-conditioning and independent likelihood audit consumers
now also honor optional fire_temperature; both scripts compile. Full Go tests,
29focused Python tests, native root/continuation/likelihood/quality proofs,
exact phase4versus2+2resume, wrong-phase-resume rejection without an output
directory, Go actor/weapon parity and diffcheck pass. No native capture remains
live. U1 remains SHA256
`21c3b680e0b6090d601d524f6da53d73ae6dd083c8e7a9a6da2e6d268ccadd26`.

An additional fixed warm/cold/U1/rules validation455 cohort now closes64more
valid objective-required games with an unchanged source fence. This follows the
viewed454 result; it is additional validation, not untouched confirmation/test.
No temperature or weights were adjusted between cohorts. Cold parent2wins,
7deaths,7timeouts,827receivedHP108selfHP; warm parent1/9/6,930received111self;
U1 3/11/2,1149received263self; rules6/5/5,774received177self. Cold parent additionally
clears bunk1case7 natively without a legacy actor-kill receipt; recorded as
secondary, not added to the primary score. The apparent fire-exploration benefit
does not replicate. Across32pairedworlds, cold6/15/11,1971received406self;
warm7/16/9,1921received299self; U1 7/19/6,2032received337self;
rules13/7/12,1276received302self. No convincing improvement or promotion.

A read-only initial score-gradient audit on the phase-weighted TRAIN bank finds
primitive parameter-gradient norms forward4.12897,side2.60756,yaw12.68513,
pitch36.96045,fire0.120079,pose0.702077. This quantifies disparate update scales;
it is not a causal quality attribution or evidence that simple rescaling fixes
combat. All native and postprocessing handles are terminal. `summary.json`
under the phase experiment root and the main fragment `handoff.json` contain
the complete current state. Preserve both original bots and the failed candidate
artifacts; the learned full-combat goal remains open.

## Common non-fire randomness and conditional root learning, November17

Evidence root: `workspace/artifacts/fragment-fire-paired-20261117-015047`.
The new `checkpoint_sampling_scope=fire_only_v1` requires an explicit restore
stream. Sampling checkpointv3 binds this scope, logical source clock, stream and
frozen model. Global-streamv2 and ordinaryv1 remain unchanged. Stream0 in the
new scope exactly matches ordinary continuation. Primitive Gaussian and pose
draws and the learned weapon draw use the original common coordinates; only
Bernoulli fire uses the derived stream seed. Nested saves preserve the scope,
and malformed version/scope combinations are rejected. A reset clears it.
The scope changes fire randomness throughout the continuation, not just at the
first decision. It does not fix future actions: different fire choices can
change observations and subsequent learned proposals.

Native smoke: an ordinary30decision restore still matches the original source
features/actions/effects; two fire-scoped streams share the root871features,
all four Gaussian latents, pose and weapon proposal while differing in fire.
An independent repeated stream exactly matches30decisions and logged native
effects. The four smoke processes are mechanics checks, not extra TRAIN worlds.
All112 new500-budget native continuations (16streams at each of seven existing
phase checkpoints) finish successfully, with at most16games at timescale2.
The Go/PowerShell source fingerprint remains fixed throughout collection.
Every one of112root observations and non-fire draws matches its family peers;
all seven roots also match the original source features under the Go verifier.

Exact native credit gives19objective clears,75deaths and18unresolved. The
admitted conditional bank contains94one-decision roots,94streams,7checkpoint
families and the same7previously used TRAIN worlds across4maps. It does not add
seven independent worlds to the prior bank. There are25sampled releases in all
112roots and22in the94admitted roots. Unresolved branches are excluded without
a fabricated zero value, leaving a selection limitation. `root-return-audit`
is descriptive only: later fire draws differ too, and missing unresolved returns
prevent calling the grouped means unbiased causal fire effects.

Fire-scoped raw fragments use `hjepa_native_fragment_v3`, with unchangedv2
transition shape; their credit targets use `hjepa_fragment_mc_targets_v2`.
Both new headers are required together with sampling checkpointv3/scope. This
matters because the old Python loader did not validate nested sampling version:
a new outer header makes it reject conditional data before training. Actual
old-snapshot rejection is verified. Pre-fit prototype files that used the old
outer headers are archived under `prototype-v2`, then the same immutable native
captures are re-exported and replayed with the corrected headers. Replaced raw
hashes invalidate prototype target bindings. No fit consumes the prototype.
The new learner also rejects mixed joint/fire scopes and requires an explicit
`--fire-conditional` objective; no implicit conversion to joint PPO is allowed.

The conditional objective uses only the Bernoulli fire likelihood ratio and
native same-root other-stream return baseline. It requires index0 only,
identical observations and common non-fire draws within each family. Full joint
primitive/pose/weapon KL remains measured and gated at mean0.02; clip0.1 and
gradient norm cap1 remain. This is a conditional root experiment, not an
unbiased full-episode joint policy-gradient claim. Runtime remains the full
learned primitive and masked weapon actor without forced releases or choices.

Two update parameterizations are declared before fitting: shared encoder/actor
with LR0.0001 and `--fire-head-only` with LR0.01, which freezes everything except
weight/bias row4 of the final actor layer. The differing learning rates are
explicit; this is not an isolated freeze-only causal ablation. Both have a fixed
budget of up to four updates. The shared arm accepts4updates;
the head arm accepts2 and rejects the next line-search proposal. The latter's
`fit-head-4` directory denotes budget4, not four accepted updates. Neither model
is selected by looking at validation steps.

Shared model SHA256
`8b3199d2306431d9a37458fefaf2b583615b970a6e918dff054c438be503b1f2`,
final meanKL0.0199692047839227/maxKL0.029557892241045082.
Head-only model SHA256
`08c41885349e9bfce14d2fa8494ced77376b89a7333c463f4507302154a64b12`,
meanKL0.0014485068114731582/maxKL0.0018374795783585376.
Actual exported head-only parameters equal the parent everywhere outside the
last fire row. Mean release probability on the seven unique TRAIN roots changes
from0.2198784381 to0.2233504000/shared and0.2422585885/head. Head-only non-fire
raw logits remain exactly equal on those roots; the parameter identity proof is
stronger than this finite input check. This says nothing yet about combat.

Go actor/weapon parity covers all seven unique roots, not eight duplicate rows
of the same family. Complete model AND optimizer/history states match shared
direct4 versus2+resume2; head budget4 versus2+the rejected next attempt also
matches exactly. The actual historical default joint1checkpoint resumes to the
same old2update model bytes, SHA256
`76ecb075c5d0f4dfe72b4516196687b146fb935ddce36c955bcc3b511de82955`.
Wrong objective, incompatible head resume, malformed scopes and common-root
draw changes are rejected. Full Go tests and31focused Python tests pass.

Fresh objective-required validation456 is frozen with16worlds on8maps and five
variants: warm parent,shared fit,head fit,U1,rules. Both final fixed-budget model
files predate viewing this cohort. Results are pending until all80games close;
no quality or promotion claim is inferred from the changed fire probabilities.

All80validation456games now close validly, with unchanged runtime source fence
and frozen model JSON; primary goals are independently joined to the native
objective counters. Warm parent2wins/10deaths/4timeouts,1079received96self;
shared3/8/5,1137received138self; head4/7/5,965received74self;
U1 9/6/1,883received181self; rules8/5/3,716received95self. There are no secondary
clearances outside the primary score on this cohort. The first apparent head
gain is two added wins, base3case5 and bunk1case6, without losing a parent win.
This is a small initial observation, not a quality gate.

An unchanged-model repeat requests32fresh worlds. Initial457/458plan generation
fails before gameplay at city1 soldier-ss seed1922182457 after128
profile_actor_overlap attempts. No native games start for that failed plan; it
is a technical generator failure, not a model loss. The next fresh459/460plan
passes the same pre-game verification. The same five frozen models then finish
160more valid objective-required games on32pairedworlds/8maps with an unchanged
source fence. Results: warm7/16/9,2196received172self; shared5/20/7,
2341received182self; head5/18/9,2217received196self; U1 10/8/14,
1336received163self; rules15/6/11,1371received26self. Living native clearances
without the legacy actor-kill receipt occur at flat case19 for warm/head/U1/rules
and are recorded as secondary only.

Combined48validationworlds/240validgames:

| Variant | Wins | Deaths | Timeouts | Received HP | Self HP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Warm parent | 9 | 26 | 13 | 3275 | 268 |
| Shared conditional fire fit | 8 | 28 | 12 | 3478 | 320 |
| Fire row only fit | 9 | 25 | 14 | 3182 | 270 |
| U1 | 19 | 14 | 15 | 2219 | 344 |
| Rules | 23 | 11 | 14 | 2087 | 121 |

Head versus parent:7sharedwins,2head-onlywins,2parent-onlywins,37neitherwins.
The initial gain does NOT replicate; total primary wins are equal. No promotion
or superiority claim. Additional validation is not untouched confirmation/test.

`root-score-cancellation-proof.json` confirms the intended finite-bank numerical
property: on the94common-nonfire roots, centered peer advantages cancel initial
non-fire score gradients to at most1.88e-14, while fire norm is0.0922597527.
`common-native-command-audit.json` verifies all112first native commands have
identical non-attack fields. This establishes controlled mechanics, not general
combat gain or an unbiased remedy for the18missing tail returns.

The48pair `head-first-difference-audit` uses Go HJEPA feature extraction, not an
approximate Python observation comparison. Before the first changed sampled
action, actual feature vectors match; in41pairs that first changed action is
fire only, and in7no sampled action differs. A second audit finds native attack
bits differ in all41 and all other command fields match. Those first differences
occur with Blaster18,HyperBlaster7,Machinegun5,Railgun4,GrenadeLauncher3,
Shotgun2,Chaingun1,BFG1; none is initially an equipped hand-grenade difference.
The two gained wins first diverge on Machinegun; the two lost wins first diverge
on Railgun/Blaster. Some changes occur during weapon recovery, so an attack bit
is not proof of a shot or the decisive causal step. Training only grenade-phase
states did not constrain changed fire behavior to grenades. This supports
checking other declared random streams and broader weapon-state TRAIN coverage,
not hardcoding a release rule or concluding a particular gun caused each loss.

The current actor stream is fixed in each frozen model; native condition seeds
vary, but that is not evidence of multiple independent actor RNG streams. A
future robustness check should freeze additional explicit primitive/weapon seed
copies with identical neural weights and compare matched streams, counting
condition worlds separately from repeated actor draws. This is proposed, not
implemented here. Then broaden common-random TRAIN roots beyond equipped
grenades before another fit. Keep viewed456/459/460 out of TRAIN and untouched
test claims. All capture, fit and postprocessing handles are terminal. Existing
H-JEPA and live U1 are preserved; the full learned-combat goal remains open.

## Frozen actor streams and broader root-action experiment, November17

`fragment-actor-streams-20261117-023959` closes128valid objective-required games
on16fresh validation461worlds/eight maps. Warm/head neural weights and fire
temperature2 remain fixed; copies differ only in explicit primitive and weapon
sampling seeds1729,2718,3141. Actual runtime JSON and source fence match.
Per-stream warm/head primary wins:3/6,3/3,4/4. Across48paired actor draws on the
SAME16worlds, warm10/head13wins; these are not48independent worlds. U1/rules run
once per world:7/9wins,3/3deaths,6/4timeouts. Both additionally clear ware1case9
without the legacy actor-kill receipt, recorded separately. The win gain occurs
only in one actor stream; do not select it as best or claim robust superiority.
No neural fit or promotion occurs in this check.

The next experiment is `fragment-weapon-roots-20261117-034744`. Before gameplay,
24TRAIN448..451conditions are selected from64candidates:three per map across
eight maps, preferring distinct initial weapons and descriptors. Initial weapon
coverage spans10types. The generic registry's learned-plan model adapter remains
PPO-only, so the frozen plan is explicitly a condition-only rules-mode schedule;
actual source workers use full learned H-JEPA with stock weapon contract. This
does not substitute rules actions for learned source decisions. Each source
model copy changes only primitive/weapon seeds to its declared engine seed.

New observed predicates `equipped_weapon_v1` and `weapon_trigger_phase_v1` accept
a canonical weapon name. They only observe public current weapon/gunframe/ammo
and hold before observation consumption or decision. The trigger predicate uses
conservative stock idle or trigger-sensitive firing phases, with ammo minimums;
a pending use request can still preempt firing, so native paired evidence remains
authoritative. Resolved captures clear both phase and weapon fields. Alias
normalization reuses the policy's equipped-weapon table. The old grenade phase
and exact-frame behavior remain supported; no current weapon is forced.

The old fire-only stream changed all future fire draws, adding noise when only
the first decision was fitted. New sampling checkpointv4/root_fire_only_v1 varies
only the first loaded fire draw. All later policy RNG coordinates are common.
A nested pre-decision save after that decision drops the expired override and
keeps the logical common future, rather than applying it again. Native smoke
proves opposite-root-action replay, exact repeated branch, exact ordinary30step
continuation and exact source30step continuation when the first action matches
despite a different root draw coordinate.

Two actual sampled root streams are selected to obtain opposite fire actions;
this is action enumeration, not IID policy sampling. Raw fragments therefore
usev4 and MC targets usev3. `--root-counterfactual` requires exactly one complete
fire/release pair per root, common inputs/nonfire proposals and actor logits.
It computes the frozen behavior Bernoulli masses, their expected-return baseline
and mass-weighted clipped ratios. Analytic two-action gradient testing verifies
the correct expected Bernoulli return derivative; no unweighted selected-sample
PPO claim. Both native tails must establish a terminal; incomplete pairs require
a bootstrap and cannot receive an invented zero return.

`--seed-equivalent-models` is explicit and requires a frozen `--reference-model`.
It permits differences ONLY in primitive/weapon sampling seeds, checks each raw
model hash and actual seed receipt, and binds reference/distribution/algorithm
to resume. Any changed neural weight or temperature rejects. A checkpoint family
cannot change its behavior model, and reserialization cannot hide duplicate
condition/root captures. Default strict admission remains unchanged.

Of24source attempts,14capture an eligible learned root with7equipped weapon types;
nine never observe the requested phase, one lacks learned ownership at the root.
These misses remain recorded, not counted as admitted TRAIN worlds or model
quality losses. Native paired continuations and fitting are still pending.

Storage audit finds only about35GBfree onF. Recent generated journals already
use NTFS compression (one checked6.29MBtrace occupies2.97MB); repeating compression
does not materially free space. No deletion or logical file-content change occurs.
The two-action common-future design avoids collecting16duplicate suffixes per
root. A logical model SHA check remains identical after the compression pass.


### Root-only action enumeration: admitted bank and CUDA fit (2026-11-17)

Artifact root: `workspace/artifacts/fragment-weapon-roots-20261117-034744`.
All28 native branches and source fences completed. Nineteen branches establish a
native terminal, but admission requires both outcomes: nine complete pairs,
18training rows, nine unique worlds/families on seven maps. Admitted weapon types
are Blaster, Grenades, RocketLauncher and HyperBlaster. The five incomplete pairs
(indices0,8,9,12,15) are excluded as whole pairs; no invented zero bootstrap.
Selection by observed trigger phase and terminal availability limits coverage.
Equal-return pairs remain admitted. See `admission.json` for exact inputs.

The frozen-parent fire-head-only candidate accepts all four CUDA updates.
SHA256: `3ceef1ae8b59a38b41370cdd991a1d21a2cbd6ebe20f74a9528044581322f5cf`.
Final mean/max analytic full-distribution KL:0.007910667793028134 /
0.014778714344790875. Every parameter outside the final fire row is unchanged.
Direct4updates equal2plusresumed2: model bytes, entire checkpoint including
optimizer, and history all exact (`resume-proof.json`). Go actor and weapon
parity pass for all nine root fixtures. Focused Python35tests and full
`go test ./...` pass. Native pilot validates source-equivalent first action,
opposite root actions, repeated branch and ordinary continuation.

Fresh validation462 is frozen at16worlds/eight maps, with three matched actor
seeds1729/2718/3141 for parent/candidate and U1/rules once per world:128games.
These are48actor draws over16worlds, not48independent worlds. Primary score
remains actor-kill receipt plus exact native alive0/pending0; other living native
clearance is secondary. No model promotion. Quality capture currently running.


Fresh462 quality completed:128/128 valid native games; source fence unchanged.
Parent/candidate wins by matched actor seed are3/4,4/4,3/2 out of16each.
Aggregate10/48vs10/48 across the same16worlds: no win improvement.
Parent/candidate deaths20/21,timeouts18/17, receivedHP3004/2995,selfHP217/260.
U1:3wins7deaths6timeouts; rules:5wins6deaths5timeouts (each16games).
No secondary-only native clearances. Opposing actor-seed effects cancel; this
small validation cannot establish general superiority over U1 or rules.
Complete optimizer/history resume, negative CLI admission checks, Go parity and
native provenance are accepted. U1 hash remains unchanged. No promotion.
Next: expand independently selected TRAIN roots and complete terminal pair
coverage, particularly missing weapon classes; then freeze a new validation.
Short-fragment mechanics work, but this9root four-weapon fit has no measured
aggregate win gain. See artifact `summary.json` and `native-quality/summary.json`.


### Expanded root bank, bounded PAK pool and second fit (2026-11-17)

Artifact root: `workspace/artifacts/fragment-expanded-roots-20261117-042836`.
Eight verified TRAIN plans480..487 yield128candidate worlds. Before gameplay,
select64worlds, eight/map, prioritizing underrepresented initial weapons, then
map/descriptor/seed. All11initial weapons are represented in the frozen selection;
this does not mean all are admitted. Existing parentcf... remains frozen.

Initial64source attempts yield31accepted roots,22phase misses, two unproved
source continuations, one native barrier mismatch, and eight runtime setup failures.
The latter are infrastructure failures, not model losses. Actual original PAK and
all32shared PAK shards reached1024NTFSlinks. With no live native processes and an
accepted source fence, expand `prepare_runtime.ps1` bounded allocator32to64.
Only this file changes between native phases. PS parsing passes. All eight exact
failed conditions start on retry in new output directories; seven yield eligible
roots, one a phase miss. The original attempts remain. Actual asset32PAK is created;
no model/game binary changes. Final38roots cover ten weapon types/eight maps.

Collect76two-action500frame continuations. At5GiBreserve, the supervisor prevents
seven unstarted jobs;69completed branches stay intact. Replace only identical PAK
copies in stopped accepted branches with shared hard links, checking hashes before
and after. No trace/save/result deletion. Resume only the seven never-started jobs.
All76branch reports accepted; all76tail exports, prefix exports and root proofs
pass.62native terminal target files yield28complete pairs/56rows; ten incomplete
roots are excluded entirely, including single-terminal branches. Do not glob all
62targets into a fit. Exact input list and hashes are in `admission.json`.
Admitted coverage: eight weapon types, seven maps; no Railgun, Grenade Launcher,
BFG or base3 in this terminal-complete fit. State/terminal-availability bias remains.

CUDA fire-head-only fixed budget4 accepts2updates; candidate SHA256:
`f289594942a37891f23b35ffdbe4b0bdc071c616ff9844cbd2aee07d123ef70b`.
Final mean/max analytic full-distribution KL0.0027992492833873984 /
0.0052458533051013695. Other model parameters equal parent exactly. Complete
checkpoint/optimizer/history/model resume is exact, including the rejected next
proposal. Native Go/CUDA actor error2.66e-15 and weapon3.55e-15 across all28roots.
First parity invocation omitted the actor-only flag and rejected the fixture;
corrected invocation passes; no runtime change made to accommodate it.

Fresh validation463 is frozen before gameplay:16worlds/eight maps, parent/new
head at actor seeds1729/2718/3141, U1/rules once/world,128games. Capture running.
No promotion. Development record and handoff point to current artifacts.


Fresh463 closes128/128valid native games; all-source fence unchanged.
Matched actor-stream wins parent/new head:2/2,0/1,0/0 out of16each.
Aggregate2/48vs3/48 on the SAME16worlds: deaths31/28,timeouts15/17,
receivedHP3341/3076,selfHP862/886. U1:3/16wins9deaths4timeouts; rules:4/16wins,
5deaths7timeouts. Other living native clearance remains secondary (parent7,
head8 including primary wins), never silently rescored as primary wins.
Small actor-seed-dependent improvement does not establish a quality gain.
All228stopped branch PAK archives are hash-verified after sharing; native saves,
traces, targets and existing versions retained. U1 remains unchanged. No promotion.

**Autonomy correction and next priority:** Retrospective living first-life trace
and exact native-objective joins expose rules ownership while opponents remain.
Example: warm1729base1case0, frame218visibility timeout then system2_noncombat,
through398while native alive1/pending0. Across48matched actor draws, parent12and
head16games have such non-provider living-combat frames. U1 has2/16, only5frames.
All these fallbacks are combat_visibility_timeout/system2_noncombat, not setup.
No primary learned win in this cohort contains these handoffs, but the overall
policy is mixed control and cannot be described as fully autonomous encounter
control. Keep all games and original primary scores; postselecting no-handoff
successes would bias the comparison. `ownership-audit.json` retains per-frame
counts/first offending frame/native counters and fallback reasons. Native counters
are diagnostic supervisor truth only, never added to model inputs.

The current H-JEPA ownership grace is30nativeframes; PPO with threat memory uses
its200frame horizon. An arbitrary longer timeout is not the desired solution.
Next implement an explicit full-encounter learned ownership option with legacy
behavior preserved, no unseen-enemy coordinates/native objective policy inputs,
and save/load scope binding. Verify continuous control during no initial contact,
long occlusion, observed defeats, life/map/reset boundaries and native replay;
then collect compatible autonomous suffixes. Earlier learned-mode results retain
their measured scores, but full-autonomy claims require this separate ownership
verification. Do not relabel historical hybrid captures as autonomous.


### Explicit full-encounter ownership implemented (2026-11-17)

Opt-in H-JEPA model header: `"ownership_scope": "full_encounter_v1"`.
Omitting it preserves the existing visible-combat/grace ownership policy; unknown
scope values reject. This header is part of the existing exact model hash, so
sampling/checkpoint restore already rejects cross-scope models and offline
seed-equivalence admission cannot merge legacy and full scopes. NN parameters,
features, sampling algorithms, engine and old model files remain unchanged.

After existing health/staleness/setup gates, full-scoped providers infer on actual
empty observations as well as visible combat, without calling Planner.command
for normal living inference. No invented targets, native objective counts or
hidden positions enter the model. Observed defeat does not silently mean the
encounter is complete: the caller determines the encounter endpoint. Death,
staleness, setup, unsupported execution and actual provider failures remain
explicitly guarded; continuous ownership still needs trace verification, not
just a mode name. This is an experimental local mode, not promoted to live co-op.

Empty engagement bookkeeping now uses lastSeen0, and checkpoint validation/restore
preserve that sentinel instead of aging/rebasing a nonexistent contact. Existing
observed-contact bounded clocks retain their checks. No checkpoint schema/model
hash bypass is introduced. Tests cover630living frames with cold search, a seen
then occluded enemy, observed defeat, safety gates,340empty pre-save frames,
exact features/actions/samples after restore, nested capture, cross-scope restore
rejection and unknown scope rejection. Focused tests run with actual base1BSP.

Artifact root: `workspace/artifacts/fragment-full-encounter-20261117-050928`.
Native source on the known base1case0 condition captures pre-decision219 with
prefix218, precisely after former visibility handoff. Source/barrier proof and
30step restored continuation pass (60independently joined native single steps).
No learned sample or usercmd is forced. The64game diagnostic compares warm/head
legacy/full variants at actor seed1729 on all16viewed validation463worlds/eight
maps. NN parameters identical within each scope pair; this is ownership and
execution diagnosis, not fresh quality evidence. Capture currently running.

Default `go test ./...` passes. With Q2_SEARCH_SCAN_ROOT pointing at original
assets, optional AAS fixtures are missing. Runtime q2go fixtures provide AAS,
but three pre-existing navigation/jump checks fail: TestBase1FarReturnFirstRise,
TestBase1SecondRiseEntryHasVerifiedShortJump, TestBase3ReturnSecondRampPlan.
A read-only Go overlay disables the new bot ownership changes and reproduces
those same failures. Logs/overlay preserved in the artifact root; do not present
this optional extended navigation suite as green or as a new combat regression.
The new full-encounter tests pass against actual BSP with both fixture roots.


Full-encounter diagnostic CLOSED:64/64valid native games,32legacy and32full-scope
on16known worlds/eight maps. Native-objective/live first-life ownership audit
finds0handoff games for both full arms (1858warm+2479head=4337live-combat provider
frames), versus5/16warm and6/16head legacy. Every post-release first-life trace
frame has a current policy observation. Native setup/health guards remain intact.
Case9 explicitly covers no initial contact at release100; base1case0 covers the
former timeout at218. No native counters or hidden targets are inference inputs.

All32legacy episodes reproduce their historical463native commands exactly:
5571commands. Within all32legacy/full scope pairs, common observations/commands
match until the first ownership boundary; NN weights and actor seed unchanged.
Independent frozen Go replay (`q2hjepa-audit --require-replay`) accepts5411native
full-scope provider proposals with zero mismatches (2409warm+3002head), including
post-clearance commands excluded from the living-objective ownership metric.
The generated datasets independently bind native applied commands.

Existing primary wins remain2/16for all four variants. Warm legacy/full deaths9/10,
timeouts5/4,receivedHP1056/1180,selfHP286/439; head deaths9/9,timeouts5/5,
receivedHP1079/1113,selfHP322/357. Continuous ownership is established for this
scope, not a combat-quality improvement. Do not promote either candidate or treat
these viewed conditions as fresh validation. Full-scope neural training is not yet
performed. Go source fence, model hashes and original U1 remain unchanged.

Next collect fresh full-scope TRAIN roots/continuations; never merge labels from
legacy-scope state/control distributions. Frozen warm neural parameters provide
the baseline; historical conditional head fits remain preserved comparisons.
Then use independent full-scope quality tests across maps and actor seeds.


### First fresh full-scope fragment TRAIN bank (2026-11-17)

Artifact root: `workspace/artifacts/fragment-autonomous-train-20261117-052625`.
Use `workspace/artifacts/current-autonomous-train.txt` as the actual root pointer
if timestamps differ. Native sources use full_encounter_v1 warm neural baseline
SHA0ea73be9c6ca8dcc255cf68bafd3d56b9dacd589e67ce30a8f2f99e42114ee9e.
Eight verified new TRAIN plans488..495 produce128candidates. Select64worlds,
eight/map, balancing initial weapon coverage before gameplay; all11initial types
present. Engine seeds are disjoint from previous TRAIN480..487selection. Per-world
primitive/weapon seeds are declared; only those two seed fields vary. No old
legacy-scope labels or viewed validation conditions enter optimizer inputs.
Full-scope source acquisition running; actual admitted coverage remains unknown.

Before new collection, `run_combat_checkpoint_branch.ps1` shares immutable PAK
archives through the existing bounded NTFS allocator while still copying mutable
runtime files. Native checkpoint runtime hashes remain verified. A30step replay
from full-scope source219 reproduces all observations/commands/effects, and all
three archive source/target NTFSfile IDs are identical. This prevents duplicate
archive allocation at branch creation, rather than reclaiming space after games.
Only PS storage preparation changed since the frozen full-scope Go binary build;
source and binary bindings record that distinction. PS parse passes; no game/NN
behavior changes. Existing source outputs and original PAK contents preserved.


First full-scope TRAIN collection closes:40accepted sources/64attempts, ten
weapons. Sixteen phase misses, five control-file atomic rename AccessDenied
errors and three unproved source continuations are retained as acquisition
failures, not quality losses. Exact source-failure diagnostics identify
`internal/harness/checkpoint/barrier.go` WriteCapture atomic replacement; reader
sharing is suspected from Windows access-denied errors. A bounded retry remains a next
infrastructure fix after the fenced epoch. No failing capture is admitted.

All80native500frame root-action branches complete. First-life ownership audit
joins native objective telemetry (held loadedframe1 uses checkpoint-bound source
objective, later frames use tail telemetry), confirms15101living-objective
provider frames and zero rules handoffs. All80tail exports/prefixes/root proofs
pass;66native terminal target files yield32complete pairs/64rows. Eight
incomplete roots excluded as whole pairs. Admitted coverage:10weapon types/all
8maps; BFG remains absent. Four equal-return pairs are kept. Terminal labels:
14native objective clears,50native deaths. No legacy-scope label mixing.

Fixed budget4 CUDA root-fire-row fit accepts2updates. Model SHA256:
`c39bded8434a8a952dc06a3261009e120a01125d80df2b58b6bd36d824bde91e`.
Final mean/max full-distribution KL0.0023887619033638487 /
0.0034814536113050784. Other parameters and full-encounter scope remain frozen.
Complete model/checkpoint/optimizer/history resume is exact; Go/CUDA actor and
weapon parity pass all32roots. Eight-plan source fence accepted unchanged.

Fresh validation464 frozen:16worlds/eight maps, three actor-seed strata for
parent/new full-scope head, U1/rules once/world,128native games. This is48matched
actor draws over16worlds, not48independent worlds. Existing primary and secondary
scores unchanged, no promotion. Native quality currently running; verify ownership
again rather than relying on the model header alone.


First autonomous fit quality CLOSED:128/128valid native games at fresh464.
Parent/candidate primary wins by actor seed:2/1,3/4,3/3 out of16each.
Aggregate8/48vs8/48 over SAME16worlds: deaths27/29,timeouts13/11,
receivedHP3471/3617,selfHP722/857. Secondary native clearance12/13 including
primary wins. U1:4/16wins8deaths4timeouts; rules:6/16wins5deaths5timeouts.
Primary score is unchanged: native actor-kill evidence matching the configured
class multiset after release, plus native alive0/pending0; other clearance remains
secondary. Known463cases illustrate legitimate environment-assisted clearance:
base3case5target_laser kills gunner,ware1case9func_train crushes infantry. Actor
kills the other monsters, but not the full class multiset. This explains absent
primary receipts; no missing public-kill-parser bug is established. Do not change
scores after seeing outcomes. Full-scope counters remain retrospective truth only.

All96H-JEPAquality games retain provider ownership on living live-objective frames;
post-release first-life capture coverage is complete. Independent Go frozen
replay accepts all six H-JEPA model/actor-stream cohorts with zero proposal
mismatches. Source fence unchanged and U1 hash preserved. No promotion, no primary
win improvement; harm metrics worsen. The current32root full-scope fire-only fit
is a verified negative result, not proof the broader fragment approach fails.

Next direction: keep warm full-scope baseline and explore/credit the joint root
action (movement, aim, pose, fire, weapon) with common future RNG and correct joint
density. Do not repeatedly increase only the fire-row sample count. Preserve
terminal/bootstrap admission: unobserved tails never become zero returns. Larger
banks first need bounded storage using immutable checkpoint data and a robust
WriteCapture atomic replacement; five observed Windows rename failures are
infrastructure exclusions and the suspected reader-sharing cause still needs a
reproducing test. Existing legacy/full models and all experiment artifacts retained.
