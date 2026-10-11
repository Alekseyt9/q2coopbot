# Shared target held-out results, 2026-10-11

Common-source pool320/320 accepted with unchanged source,16 slots×2,
wall860.16 seconds. Four controllers each ran the same80 held-out fixtures
and test seeds across20 recipes, four maps, pair/trio groups and two inventory
families. The native comparator checked actual runtime files/binaries, source,
reset/environment settings and generated fixture equality. This is not proof
of complete internal AI/world-state equivalence.

| Metric | Frozen update9 | Trained control10 | Shared-target10 | Rules |
|---|---:|---:|---:|---:|
| Goals/80 | 33 | 30 | 28 | 23 |
| Kills | 120 | 102 | 105 | 112 |
| Deaths | 40 | 47 | 48 | 49 |
| Damage dealt | 23496 | 22959 | 23034 | 24191 |
| Damage received | 5452 | 5485 | 5631 | 5357 |
| Recorded frames | 10103 | 8674 | 8937 | 8824 |

| Map,20 cases/controller | Frozen goals | Control goals | Shared goals | Rules goals |
|---|---:|---:|---:|---:|
| base3 | 7 | 5 | 6 | 5 |
| city1 | 9 | 10 | 12 | 9 |
| jail1 | 1 | 2 | 2 | 3 |
| ware2 | 16 | 13 | 8 | 6 |

The shared candidate is not promoted. It loses five goals versus the frozen
parent and two versus the equally trained control. Ware2 is the main
regression. Fewer recorded frames cannot be called improved efficiency when
more episodes end in death. Both trained candidates lose to the frozen parent
overall; one further trio-only update has not established a useful architecture
gain. City1 improves, but that does not justify a global replacement.

Frozen update9 remains the experimental training reference. This80-case result
also does not establish universal superiority over rules: rules has more kills,
less received damage, and still does better on jail1.

Authoritative artifacts under `workspace/artifacts/`:

- `combat-shared-target-eval-pool-v1-20261011/report.json`;
- `combat-shared-target-vs-parent-quality-v1-20261011.json`;
- `combat-shared-target-vs-control-quality-v1-20261011.json`;
- `combat-shared-target-vs-rules-quality-v1-20261011.json`.

CUDA verification completed for all240 learned evaluation captures:9735
eligible rows for the frozen parent,8372 for the control and8579 for shared
target,26686 total. All three processors are complete without training.
Selected-target/movement/first-attack audits also completed for all three
controllers. Closed native exports were compressed with unchanged logical
SHA checks. No evaluation rows are used to train or tune weights.

| Composition/inventory | Cases/controller | Frozen goals | Control goals | Shared goals | Rules goals |
|---|---:|---:|---:|---:|---:|
| Trio/weapons | 32 | 11 | 6 | 6 | 0 |
| Trio/weapons-ssg | 32 | 12 | 14 | 12 | 11 |
| Pair/weapons-ssg | 16 | 10 | 10 | 10 | 12 |

The regression is concentrated in ordinary-weapon trios; pair goals do not
fall in this sample. This does not identify its cause. A controlled next
experiment forks both original training checkpoints into target-only scope,
reuses each branch's own accepted train-only64-case corpus, and freezes attack,
weapon, movement and aim parameters. Held-out rows remain excluded. Fresh test
offsets12:16 will compare these two candidates with the unchanged parent,
both previous decision-scope candidates and rules on matching fixtures. The
shared branch still trains its per-enemy
residual; its inactive spatial branch stays frozen.

Paired goal flips for shared decisions against its parent:8 previously lost
fixtures become wins,13 previously won fixtures become losses,20 stay wins and
39 stay losses. This is the same five-goal net regression; it is not a uniform
failure across every encounter and is not a statistical superiority test.
Receipt: `combat-shared-target-paired-goal-strata-v1-20261011.json`.
