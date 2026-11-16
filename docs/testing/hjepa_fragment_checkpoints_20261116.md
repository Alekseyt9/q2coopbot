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
