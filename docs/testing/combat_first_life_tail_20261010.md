# First-life tail accounting and next termination change

`scripts/report_combat_life_tail.py` ran on the sealed480-fight
`workspace/artifacts/movement-ppo-eval-v2-20261010` corpus. The report pins the
protocol, native member proof and each capture report; validates seed, row
accounting and member hashes; and stores `first-life-tail.json`. It reads export
metadata, without neural execution or changing rewards.

| Variant | Exported rows | Rewarded rows | Masked after first life | Masked fraction | First-life deaths |
| --- | ---: | ---: | ---: | ---: | ---: |
| Instant PPO1 | 11651 | 8119 | 3401 | 29.19% | 20 |
| Instant PPO2 | 12679 | 7918 | 4640 | 36.60% | 24 |
| Postmove PPO1 | 11706 | 6241 | 5338 | 45.60% | 29 |
| Postmove PPO2 | 11984 | 6878 | 4988 | 41.62% | 26 |
| FireBC | 11814 | 6899 | 4801 | 40.64% | 26 |
| Rules | 6455 | 4210 | 2094 | 32.44% | 9 |

Masking also has other reasons; all masks plus available reward rows were
checked against exported row totals. These fractions describe exported
transitions, not wall-clock savings or every native server frame. Dead intervals
and later lives contribute to the discarded tail. Preparation, startup and
export overhead remain even when gameplay stops earlier.

A concrete check on Postmove PPO2, case1 seed710028:300 capture game frames,
268 exported rows,61 available reward rows,207 `after_first_life` masked rows.
The last available row is step61 and already contains the death component-5.
The existing assembler marks an adjacent same-life observation with nonpositive
health as terminal `observed_death`; reward verification requires matching
native death evidence. This single example demonstrates a retained death
transition, not proof for all proposed early-stop paths.

The same lightweight report is queued after the current miss-objective
evaluation's firing reporter. Receipt:
`workspace/build/miss-ab-life-tail-process.json`; its progress/output are inside
`workspace/artifacts/miss-reward-ab-eval-20261010`.

Next implementation, after the source-bound A/B evaluation is terminal:

1. Extend supervisor stopping alongside the existing goal stop. Require the
   released generation/actor, native first-life player death and a complete
   observed transition closing the death window. Stale health, prep deaths,
   another actor or a reused generation must not trigger it. Keep native death
   evidence outside the policy observation.
2. Store a separate death receipt and distinct termination reason. Accept a
   shorter frame budget only after verifying that receipt. A failed capture,
   process timeout or undecoded trace is not an early-stop loss.
3. Reuse the Go client's existing configured stop-file mechanism, after the
   death observation is written. Export and preserve the actual death terminal
   with its full reward; do not truncate the preceding living action or relabel
   death as goal completion. Terminal bootstrap remains zero.
4. Add focused non-neural boundary tests and a native smoke covering a live
   goal win, early death, no death before maximum, wrong actor/generation and
   missing/late observation. Compare first-life reward/terminal accounting to a
   complete captured death trajectory, not just worker exit codes.
5. Use the same stopping policy for every arm of subsequent matched captures,
   then measure wall time and allocated storage. Preserve existing sealed
   captures and the currently running common protocol.

At the time of the initial report, implementation was pending while the
480-fight A/B evaluation held its source binding. That evaluation is now closed.

A supervisor receipt prototype is staged only under ignored workspace/build/combat_death_stop.proposed.ps1, outside active source binding. On the closed Postmove seed710028 native log, both death and first nonpositive observation are server frame159; the receipt must allow this equality rather than requiring an extra frame. A native replay plus negative checks for living actor, later life, missing native death, wrong generation and a pre-release death passed. This is a receipt candidate, not runtime early-stop acceptance: ordered native death-window proof, full terminal/reward verification and short-frame-budget validation still have to be integrated before rollout. Current capture sources remain unchanged.

## Runtime acceptance

The prototype has now been integrated into registered captures. The supervisor
writes a separate first-life death receipt and uses the existing client stop
file. The exporter independently verifies the actual adjacent observed death
transition, ordered native damage window, actor/generation, receipt observation
and full death reward. A shorter frame budget is accepted only with this proof.
No terminal is invented from a supervisor receipt alone.

The first 16-case live attempt retained seven invalid captures: generated
legacy fixtures omit `map`, and the new verifier used that absent field instead
of the existing resolved capture map. This was fixed without changing the game
or policy. Failed records remain under
`workspace/artifacts/death-stop-native-smoke-20261010`.

The repeated smoke under `death-stop-native-smoke-v2-20261010` completed with
16/16 valid captures and zero pool errors, with unchanged capture source binding.
Five native first-life deaths exercised the new stop path: seeds610125,610126,
610127,630125,630126, ending after171,61,56,70,44 game frames respectively,
rather than the300-frame budget. Every death has an exporter
`death-stop-verification.json` with a verified actual terminal and death reward.
The replay audit also preserved the old complete trajectory's rewards byte for
byte. These are stopping/contract checks, not evidence of better combat quality
or a measured overall speedup.

Acceptance: `workspace/artifacts/death-stop-native-smoke-v2-20261010/death-stop-acceptance.json`.
Member proof and life-tail report are in the same root. Focused Go tests cover
actor/generation, window bounds, incomplete transitions and other rejection
boundaries without executing neural inference.

Next fresh collection uses all20 canonical families, train offsets128..131,
80 stochastic own-policy fights and the unchanged control objective, resuming
control CUDA update3. Driver: `scripts/run_combat_first_life_update.py`.
It requires the closed smoke proof and sealed parent checkpoint, uses16 slots
at timescale2, and requests `--cuda-only-export --cuda-batch-finalize` for the
full processing/training cycle. Actual CUDA completion and paired quality
comparison remain pending; older consumed corpora are not reused for training.

The real early-death seed610126 was also passed through native CUDA-only PPO
export and the CUDA finalizer. It contains44 eligible transitions and exactly
one terminal at frame154. The terminal preserves the complete reward score
-3.8232327348012776, including death component-5, is not truncated, and has
`bootstrap_zero=true` and `next_value=0`. CUDA likelihood/value/context checks
passed. Proof: `workspace/artifacts/death-stop-cuda-audit-20261010/death-terminal-audit.json`.
This verifies one live death through the training export path; it is not itself
a PPO update.

A320-case comparison is queued behind the update process with a retained
Windows process handle. `scripts/run_combat_first_life_evaluation.py` requires
sealed weights/checkpoint/report and exact CUDA checkpoint restoration before
capturing parent, updated actor, FireBC and rules on identical20-family
validation32..35. All arms use common first-life stopping and reward.
Quality, first-shot delay, movement, selected-target aim, Blaster contacts and
life-tail diagnostics follow strict native member proof. Results are pending.

The80-fight collection completed with zero invalid members and unchanged source
binding. Initial processing stopped before any PPO update: case15's offline
replay used the legacy `game_frame_limit` reason, producing different trailing
step metadata for a death-stopped trajectory. A single-death audit alone had
not exposed this difference. `cmd/q2ppo-data` now reads the typed death receipt,
rejects conflicting goal/death conditions, replays with the actual death-stop
reason, and hashes/compares the original and replayed terminal/reward proof.
The previously failing four-member case15 now exports exactly, including two
terminal transitions. Focused non-neural learningenv tests passed again.

Original capture and failed processing/evaluation receipts remain preserved.
Recovery reuses only the complete native collection; no completed update exists
in the failed processor. New processing root:
`workspace/artifacts/first-life-update4-process-v2-20261010`.
`first-life-update4-recovery-20261010` binds the old failure receipts, repaired
exporter source/binary and successful processing receipt when available.
The replacement320-case evaluation under `first-life-update4-eval-v2-20261010`
waits for that recovery process and reads the explicit recovery receipt.
CUDA batch finalization, update4 and quality comparison are still pending.

Recovery completed: all20 CUDA corpora finalized; merged rollout contains5695
eligible transitions,15 terminals and85 zero-bootstrap boundaries. Serialized
audit confirms every terminal is non-truncated with zero next value.
CUDA update4 accepted10 actor steps (40 total), preserving the original
reward/config/anchor/bank contract and resuming the sealed parent Adam state.
Weights SHA256:39b8afa944986c65a1a20c8be8773461f143b0526ed63847fc9b2ce673a68de8.
Exact CUDA actor/value/std and optimizer restoration passed. Recovery execution
receipt pins completed processing report
b03a8ade239921f592dbc89c2e8a3f685b5bd2d4373129d786308c510902243d.

The originally queued validation32..35 exceeded the registered32-seed cohort
and failed during compilation, before native evaluation. The evaluator now
checks every family's split bound before compiling. Actual replacement
evaluation: `workspace/artifacts/first-life-update4-eval-v3-20261010`,
validation24..27,320 fights,16 slots x2. It uses the closed recovery receipt and
sealed CUDA update; no training was repeated. This remains development
validation, not untouched final-test evidence. Quality results are pending.

The sealed80-fight pool efficiency report found a runtime limitation not
covered by the initial16-case smoke:15 first-life deaths, but only8 death-stop
receipts. Seven captures reached300 frames after the first observed death;
the supervisor's last-line polling skipped their short death interval before
respawn. Reward masking remains correct, but1310 exported rows are still masked
after first life. The8 stops left1610 configured budget frames unrun. This is
a budget counterfactual, not measured wall-time savings. The actual pool wall
time was198.52seconds for80 captures, including preparation/export overhead.
Different seeds preclude a causal comparison with earlier collections.
Evidence: `first-life-update4-recovery-20261010/efficiency.json`, generated by
`scripts/report_combat_first_life_efficiency.py` from pinned selected members.

Current320-case evaluation remains source-bound and unchanged. A replacement
trace cursor is staged outside active sources under
`workspace/build/combat_trace_reader.proposed.ps1`. It reads every new complete
JSONL line in order, keeps incomplete bytes (including split UTF-8 characters),
does not re-emit consumed rows, and rejects trace truncation. Closed-trace audit
recovered all7 missed first-life death observations while consuming entire
traces, including later respawn rows. Partial-byte appends, multibyte text,
repeat reads and truncation rejection passed. Receipt candidates must be kept
until native death telemetry is available; they do not replace ordered exporter
terminal/reward proof. Evidence:
`workspace/artifacts/combat-trace-reader-audit-20261010/report.json`.
Supervisor integration and live acceptance await completion of the existing
source-bound evaluation; no full coverage or additional speedup is claimed.

A staged supervisor integration also passed7/7 missed-death traces. A mock
process exits only after receiving the real stop marker. The complete real
trace is already available through later respawn rows, but native damage events
are withheld until the next poll with no new observations. The supervisor keeps
the earliest first-life death observation and emits the death receipt after
native evidence arrives; none of these losses becomes a goal win. Proof:
`workspace/artifacts/combat-supervisor-cursor-audit-20261010/report.json`.
Staged function: `workspace/build/combat_goal_stop_v2.proposed.ps1`.
This is a replay/mock synchronization check, not live UDP acceptance. The
running320-fight comparison continues with its original capture code.

The320-case comparison is now closed, with all diagnostics complete. Update4
won46/80 versus parent48/80, FireBC51/80 and rules69/80. No promotion.
See `combat_first_life_update4_results_20261010.md`.
After authoritatively terminal evaluation, the cursor and retained death
observation were integrated into active supervisor sources. Production parser
and supervisor replay checks passed7/7 again. A20-fight native regression on
the five previously implicated families is running at16 slots x2 under
`cursor-death-stop-regression-20261010`. It replays train128..131 conditions
with frozen parent3 solely for stopping acceptance; these already used
conditions and new observations do not enter PPO training. Live coverage
acceptance remains pending.

Live cursor regression is complete:20/20 valid native records, zero pool
errors, all12 observed first-life deaths have verified early-stop receipts and
actual terminal/death-reward proof. Every death ended before300 frames
(56..131frames). None of the2463 exported rows is masked after first life.
All seven previously missed seeds were exercised successfully. Evidence:
`workspace/artifacts/cursor-death-stop-regression-20261010/death-stop-acceptance.json`
and the sealed member proof/life-tail report in that root. The replayed
conditions remain excluded from further PPO training. This proves stopping
coverage for this regression sample, not quality improvement or a general
wall-time speedup.
