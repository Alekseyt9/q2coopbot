# Native PPO export and CUDA numerical verification

The new `q2ppo-data --cuda-only` path retains the native re-export proof:
`steps.jsonl`, `rewards.jsonl`, and `server_outcomes.jsonl` must match the capture
byte for byte. It also verifies fixed release/RNG, seed and model bindings,
sampled action conversion, goal stops, and source hashes. It exports exact
`Step.Next` features, next-context reset flags, and terminal/kill-handoff zero
bootstrap flags. It does not call `Review`, `VerifyMemory`, or `ValueAfter`.

The intermediate contract is `combat_ppo_native_cuda_pending_v1` and the data
file is `native-rollout.jsonl`. This is deliberately rejected by the existing
merger and trainers. No provisional zero critic values can become training
targets through that interface.

`scripts/finalize_combat_cuda_rollout.py` requires CUDA with no CPU fallback.
It checks behavior likelihood, critic values and full provider memory context
against the recorded runtime samples, then computes bootstrap from exact next
features. Terminal and kill-handoff bootstrap is zero; nonterminal time limits
retain bootstrap. Missing frames/life changes reset next memory. The resulting
ready corpus retains all native source hashes and an additional CUDA receipt.
Only then does it expose `combat_ppo_rollout_v1` to the existing merger/trainer.
CPU work remains for native proof, feature extraction, file IO and serialization;
the game companion continues to run the policy in Go during native play.

## Verification

On existing sealed records, without executing Go numerical network replay:

| Recorded branch/case | Rows | CUDA likelihood max error | CUDA value max error | Bootstrap max difference from recorded legacy export |
| --- | ---: | ---: | ---: | ---: |
| Postmove attention, solo Blaster | 404 | 5.29e-5 | 5.30e-6 | 4.23e-6 |
| Instant attention, mixed Machinegun | 362 | 3.72e-5 | 6.08e-6 | 3.42e-6 |

Features, samples, rewards, identities, terminal/truncation flags and
interventions were preserved exactly for all 766 rows. Twelve terminal/handoff
bootstrap flags were exercised. These batches contain no next-context reset
rows; the separately recorded CUDA prefix/reset/window boundary audits still
cover synthetic reset inputs. This is not native acceptance of every network
architecture or every reset scenario.

The pending contract was rejected by the merger without creating an output.
Focused non-neural merger tests passed. The production processing driver uses
this path only with `--cuda-only-export`; legacy processing remains available
for existing frozen historical receipts.

## Fresh movement training

`scripts/run_combat_movement_ppo.py` collects 160 fresh battles with the updated
wall movement guard: 80 each for Instant/Postmove attention, train offsets
112..115, 20 existing campaign/arena families, 16 refill slots, timescale 2.
It continues the sealed CUDA PPO update1 checkpoints, including Adam state,
without changing their objective/config or reusing consumed rollouts. The new
processing path explicitly selects CUDA-only export/finalization and training.
The whole actor, critic, standard deviations and spatial branch remain trainable.
Paired native validation must compare old/new weights under the same guard;
the guard-only 16-battle probe is not evidence of model improvement.

## Completed fresh CUDA update (2026-10-10)

The fresh movement capture sealed 160/160 battles with unchanged sources.
`movement-ppo-process-v1-20261010/report.json` is complete. Instant used 5406
eligible transitions and Postmove 6715; both accepted ten additional actor
steps, reaching PPO update2. CUDA training took 12.84 and 4.72 seconds,
respectively; collection and numerical verification are separate costs.
Both checkpoint CUDA audits verified exact actor/value/std and Adam state
after reload. No Go numerical neural replay was executed in this processing path.

Paired evaluation `movement-ppo-eval-v2-20261010` uses validation offsets
28..31, six variants, 480 battles, the same movement guard, 16 refill slots
and timescale2. Evaluation remains unsealed while running. One signon failure
(job116, seed1095028) produced an empty trace after server configstrings
buffer overflow; this is infrastructure failure, not a model loss, and needs
a preserved retry before a complete quality comparison can be claimed.
