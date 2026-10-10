# Confirmed miss reward experiment — 2026-10-10

Proposed independent objective: -0.02 per confirmed Blaster geometry/sky
ending, retaining positive monster damage and +5 native kill bonus. Existing
reward/config/checkpoints are unchanged. This is an A/B hypothesis, not an
improved policy claim. Machinegun has no verified per-shot denominator yet.

`scripts/audit_combat_miss_reward.py` joins exact ordered native command windows,
launch/end projectile IDs, generation, actor, entity and damage contacts. Only
alive first-life matched exclusive windows with no recovery commands qualify.
Unresolved/freed projectiles and damage contacts cannot produce miss penalties.
A miss is assigned at its impact/end window, not when a projectile is still
flying. Missing alive first-life end windows are excluded; no cross-life credit.
Future server outcomes remain training labels and never policy inputs.

On sealed480 development evaluation: Instant PPO2 has253 confirmed misses,252
creditable (-0.063 proposed mean reward per80 episodes); Postmove PPO2 has211
creditable (-0.05275); rules9 (-0.00225). These totals are not gameplay results
under the proposed reward. The audit includes the release boundary logical98
and excludes preparation via native exported commands; older hit diagnostic
uses frame>100 and therefore has a different launch denominator.

Implementation gates:

1. Add a versioned native projectile ending contract and strict delayed-window
   join to the reward exporter, preserving old objectives byte-for-byte.
2. Regression checks: delayed hit never a miss; live/corpse damage not geometry;
   unresolved/freed no penalty; exactly once; generations/actors/lives isolated;
   malformed/contradictory telemetry rejected; no-shot command has no cost.
3. Verify a native16-battle smoke with actual launch/end/damage receipts.
4. Fork the same actor weights into control and miss-cost training. A changed
   objective cannot use the current PPO resume seal as if unchanged: explicitly
   reset/recalibrate critic/Adam and record the objective transition. Train CUDA
   only, equal fresh seed budgets, native pool16 at timescale2.
5. Compare native wins, deaths, outgoing damage, time-to-kill, shots, confirmed
   misses, unresolved fraction, no-fire time and ammo. Lower shot count alone
   cannot justify promotion. Retain death/time/kill incentives to avoid inactivity.

Audit evidence: `workspace/artifacts/movement-ppo-eval-v2-20261010/
miss-reward-audit.json`. Existing trained weights have not received this cost.

## Implemented contract and current verification

`combat_reward_v5` inherits the existing damage/kill/aim/spacing objective and
adds bounded negative `blaster_miss`. Native projectile lifecycle records are
joined to ordered command windows, with exact generation/actor/entity/shot
matching and contradictory damage contacts rejected. First-life launch and
alive impact windows are required; duplicate endings cannot earn repeated
costs. The cost is emitted at impact, never while a projectile is flying.
Old reward versions neither require telemetry nor serialize the new fields.

Focused non-neural learning-environment and registry checks passed. Re-export
of two closed native battles verified byte-identical legacy steps, outcomes and
rewards. New-v5 steps and existing reward components remain exact; only the
verified miss component/score changes. This is native trace contract replay,
not a Go neural test. Receipt: `miss-reward-export-audit-20261010/report.json`.

`fork_combat_architecture_objective_cuda.py` prepared control/miss branches from
sealed Postmove PPO2. Actor/std are exactly preserved on CUDA; critic head and
residual outputs are zeroed and both Adam states reset identically. The two
weights JSON files have the same SHA256. Config/checkpoint objectives differ
explicitly. No fresh PPO update has occurred. Receipts live under
`miss-reward-ab-preparation-20261010`.

Native-v3 smoke had13 complete members and3 incomplete wall-watchdog captures
(284/290/292 of300 requested frames), all16 generated native reward-v5 data.
Incomplete captures remain failed and are not training-ready. Compile-only v1
and v2 failed before any battle (reward recipe allowlist, then stochastic
training contract respectively); original logs retained.

The synchronous scenario watchdog is now120 seconds instead of60; frame
budget300, timescale2 and goal-stop behavior are unchanged. A fresh native16
smoke `miss-reward-native-smoke-v4-20261010` is running. Do not claim complete
native acceptance or start the A/B training until its strict member proof is
complete. Raising the watchdog does not claim improved throughput. First-death
stop remains a separate planned optimization.

## Native acceptance and matched A/B started

Native-v4 smoke sealed16/16 members with unchanged sources and120s watchdog.
`miss-component-verification.json` checked1386 available reward rows:34 costs
exactly equal native confirmed-miss ending counts. This verifies reward
accounting, not a trained-policy improvement.

`miss-reward-ab-capture-20261010` collects160 fresh battles,80 per control/miss
branch, train offsets120..123, twenty families, one16-slot refill pool at x2.
Both initial weights files have identical hashes. All80 paired generated
instances and engine seeds were compared exactly. The draft miss registry
increments episode revision while explicitly preserving the original generator
seed revision, so changing reward cannot change the generated fight.

`process_combat_architecture_pool.py --model-id` selects one binding from a
shared pool, pins original pool/model receipts and records original-to-local
plan indexes. This allows independent objective/config/resume checkpoints
without recollecting fights. Each branch uses CUDA-only numerical verification,
bootstrap and PPO; both critic/Adam resets were already verified on CUDA.
Invalid native capture retries retain originals and reuse the source-bound
binary bundle. Complete gameplay losses remain valid and are never retried.
The matched training updates are pending; selector processing still needs its
live two-branch acceptance on this collection.

`miss-reward-ab-eval-20261010` is queued on the retained upstream process handle.
After both sealed CUDA updates, it compares control and miss branches before
and after, FireBC and rules:480 battles,80 per variant, validation20..23. All
evaluation variants use the original common reward/guard. Identical before
weights are repeated to measure native repeat variation. These are reused
development validation seeds; final test and promotion remain deferred.

Live update: capture completed160/160 with zero errors. The control branch completed its CUDA update (6189 eligible transitions,10 accepted actor steps, cumulative update3); the miss branch is still finalizing its20 native corpora on CUDA. No paired quality result yet. First-attack diagnostics are queued after the sealed480 evaluation; see combat_first_attack_20261010.md for the completed predecessor-corpus measurements.

Both updates are now sealed complete: control6189 eligible transitions and miss6156,86 sequences each,10 accepted actor steps each, cumulative update3. Both CUDA checkpoint audits passed (actor/value/std and optimizer state exact). Root processing report SHA256:07b77bee0f366b46d827b526739cdbd72155d640ccc3b502d09ad726c9793477. The paired480 evaluation has entered paired_native_evaluation; quality and promotion remain unverified. The small transition-count difference is reported rather than claiming identical executed trajectories.
