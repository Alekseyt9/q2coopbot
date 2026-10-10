# Target-head learning audit and controlled fork — 2026-10-10

## Measured imbalance

`audit_combat_head_learning_cuda.py` replays real pinned histories on CUDA and
verifies old behavior likelihood/memory against captured values. The audit
computes exact KL(before||after), including every available target/aim-mode
branch weighted by the old policy. Its component sum is checked against
`combat_sequence_retention.policy_kl`; this is diagnostic, not a new training
objective. Masks exclude unavailable target and weapon outputs.

On the 3118 train rows used for the weapons group update9:

| Head | Mean exact KL |
|---|---:|
| Movement | 0.000404859 |
| Aim | 0.004750403 |
| Target choice | 0.000000750 |
| Attack | 0.000000005 |
| Weapon | 0.000000028 |
| Vertical | 0.000000359 |
| Aim mode | 0.000000649 |

Movement+aim consume over99.9% of measured policy change. Mean target total
variation is0.0005194 (0.0519 percentage points of redistributed probability
mass), argmax changes0%. These observations support trying a separate target
update; they do not prove that the optimizer is the sole cause of bad combat.
Audit: `workspace/artifacts/combat-group-weapons-head-audit-v1-20261010.json`.

## Projected target-only fork

`ppo_recurrent.py --actor-head-scope target --fork-head-scope` supports an
explicit fork from a normal resumed checkpoint. Reward, rollout, critic,
architecture and native feature/action contracts remain the same. The default
all-head mode is preserved. Scope and effective learning rate are persisted;
changing scope on resume requires explicit `--fork-head-scope`.

Only rows20:29 of actor output projections train. Encoder, memory/attention,
spatial branch, other action rows and log_std are frozen. Frozen weights and
Adam moments are restored after every proposal because zero gradients alone
do not stop old momentum. This projection also runs during line-search
fallbacks; rejected steps restore the complete optimizer snapshot.
The shared Adam scalar clocks for target-bearing tensors advance; other-row
moments remain unchanged. This limitation is explicit when later unfreezing.

Target-only nominal learning rate0.003, existing joint KL limit0.005; guarded
line search backtracks proposals. This is a controlled optimizer-scope change,
not an increase of the global policy-change limit or a new switch penalty.
Conditional aim/movement outputs are fixed; actual commands can still change
when the policy selects a different enemy.

CUDA regression tests check masked-head KL factorization, unchanged frozen
outputs/momentum and complete rollback of rejected optimizer proposals.
`scripts/test_combat_target_head_scope.py`: two tests passed on CUDA.

## Training evidence

Fork from group update8, using exactly the same32 train fights/3118 rows as
the joint update9. That rollout was not consumed by the parent checkpoint.
No validation or test data enters gradient training. Four actor steps accepted;
1170 scalar actor parameters train. Critic receives its usual40 updates.
Output: `workspace/artifacts/combat-group-target-only-update-v1-20261010/`.
Weights SHA `b9f8b2bc20dea59ed1c1bc8f6590fc9494c0a9d7160db6f990fc72242c73c02b`;
checkpoint SHA `2c04c93625d9cbd1331d0625854dd7df1accfefed46e4eb54b45154b4a9115c0`.

CUDA replay audit confirms zero KL on all non-target components and target
KL0.004981492. Target total variation0.0408235 (4.0824 percentage points),
about79 times the joint update's target change. Argmax still changes0%, and
the mean probability of the previous visible target is essentially unchanged.
Larger head change is not proof of better target prioritization or combat.
Audit: `combat-group-target-only-head-audit-v1-20261010.json`.

The candidate must pass the same32 frozen held-out fights as the parent and
joint update: weapons/SSG groups on base3/ware2/jail1/city1, validation offset4.
Pool16, timescale2, separate seeds. No live/public model promotion is implied.

## Paired result and verification

All three variants share the same initial fixtures,32 seeds, frozen client,
exporter and reward SHA. Target-only captures:32/32 usable; CUDA verification
completed for all32 cases/3266 rows, with no validation training.

| 32 held-out battles | Parent8 | Joint9 | Target-only9 |
|---|---:|---:|---:|
| Group goals | 9 | 9 | 8 |
| Kills | 41 | 43 | 38 |
| Deaths | 23 | 23 | 24 |
| Received health damage | 2384 | 2288 | 2379 |
| Game frames | 3369 | 3232 | 3401 |

Jail1 remains0/8 successes. The target-only branch is not promoted: larger
target-policy change did not improve this comparison. Comparisons:
`combat-group-target-only-quality-v1-20261010.json` (parent vs target),
`combat-group-joint-vs-target-quality-v1-20261010.json` (joint vs target).
This sample does not establish that separate-head training is always worse.

Full saved-checkpoint audit on CUDA additionally proves exact preservation of
all frozen actor weights, log_std and their Adam state. The four target-bearing
tensors have unchanged non-target rows/moments and their Adam clocks advance
exactly four steps. Receipt:
`combat-group-target-only-checkpoint-audit-v1-20261010.json`.

Registry CUDA processing now forwards `--actor-head-scope`, `--target-head-lr`
and `--fork-head-scope`; normal training remains the default. Existing
architecture-resume integration was moved to CUDA for generation of test
likelihoods too, and passed two consecutive default-mode updates for GRU,
temporal attention and entity attention. Target-aware scope applies to the
supported GRU/temporal target-head contracts, not legacy entity actors without
that head. CPU checkpoint inspection/serialization is not neural inference.

Next training experiment should use more successful group transitions and a
curriculum from two to three ground enemies, while retaining separate weapon
and geometry checks. Do not tune a switch penalty or learning rate repeatedly
on these same32 evaluation seeds. Continue from the existing baseline lineage;
keep this failed candidate and its proof artifacts for comparison.
