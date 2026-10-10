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

Implementation is pending. Current Go/PowerShell/native sources are kept
unchanged while the480-fight A/B evaluation runs; this report does not introduce
runtime death stopping.

A supervisor receipt prototype is staged only under ignored workspace/build/combat_death_stop.proposed.ps1, outside active source binding. On the closed Postmove seed710028 native log, both death and first nonpositive observation are server frame159; the receipt must allow this equality rather than requiring an extra frame. A native replay plus negative checks for living actor, later life, missing native death, wrong generation and a pre-release death passed. This is a receipt candidate, not runtime early-stop acceptance: ordered native death-window proof, full terminal/reward verification and short-frame-budget validation still have to be integrated before rollout. Current capture sources remain unchanged.
