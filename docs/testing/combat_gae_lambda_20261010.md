# Controlled GAE lambda comparison (2026-10-10)

## Question and controls

Does longer reward credit assignment help the current Temporal Attention
combat actor finish group fights? Gamma remains 0.99; GAE lambda changes
from 0.95 to 0.99. At 10 game ticks/sec, the fixed-bootstrap kernel for a
TD residual five seconds later changes from 0.0466 to 0.3660. Learned value
estimates also carry future effects, so this is not a claim that the old
policy ignores late rewards.

Both updates start from the same group update8 actor, critic, Adam state,
RNG state and consumed-data history. Both use the same 64 own-policy pair
training captures / 7522 rows, with the same architecture, reward, learning
rates, KL limit and ten accepted actor steps. These captures were generated
by the shared parent, rather than either newly updated policy. No validation
capture is used for training.

`scripts/fork_combat_gae_checkpoint_cuda.py` permits only a lambda change;
the normal resume config check stays strict. An explicit fork record is
preserved in later checkpoints and reports. Saved state identity is checked
on CUDA, including optimizer and RNG tensors; model JSON bytes are copied
unchanged. CPU deserialization is serialization work, not neural inference.
The metadata rejection test rejects gamma, reward, learning-rate, extra-field
and invalid-lambda changes.

## Artifacts

All paths below are relative to `workspace/artifacts/`.

| Variant | Weights | SHA256 |
|---|---|---|
| Shared parent, update8 | `combat-group-training-processing-v1-20261010/update/weights.json` | `47fe9ecfe1cda616656c2000cae05e6dee8d91c424f9aaff4d358b727d2e536d` |
| Lambda 0.95, update9 | `combat-pair-curriculum-training-processing-v1-20261010/update/weights.json` | `bfc02feaaafa6e92164da8fc32e1aeb7f6085e95705a7034222753855c2276f3` |
| Lambda 0.99, update9 | `combat-gae99-pair-update-v1-20261010/weights.json` | `4bf041db02411645725a55538270acc2ddb4914e311f55b91e547d781532508f` |

Fork proof: `combat-gae99-parent-fork-v1-20261010/report.json`.
Config: `combat-gae99-training-config-v1-20261010.json`.
Training corpus: `combat-pair-curriculum-training-processing-v1-20261010/merged`.
Lambda 0.99 training report: `combat-gae99-pair-update-v1-20261010/report.json`.
Training and numerical policy verification run on the RTX 5070 with CUDA.

The two training reports have identical behavior, rollout, sequence, anchor
and bank hashes, row counts and actor-step counts. Trainer-file hashes differ
because the new version preserves the explicit config-fork history; the
training algorithm is unchanged apart from the supplied lambda.

CUDA audit on the 7522 training rows:
`combat-gae99-pair-head-audit-v1-20261010.json`. Exact policy KL is 0.00481;
aim contributes 0.00435 and movement 0.000439. Target mean total variation is
0.000244 (0.0244 percentage points); target argmax never changes on these
histories. Old likelihood and recurrent-state replay pass their tolerances.
Longer credit assignment still produces most policy change in aiming and
movement. This is a training-history diagnostic, not proof of quality gain.

## Held-out evaluation

`combat-gae-ab-pool-v1-20261010` mixes three plans in one immediately refilled
16-slot pool, timescale2. Plan indices: 0 parent, 1 lambda0.95, 2 lambda0.99.
Each has 48 validation fights, count4 / seed offset12. Identical cases and
seeds across variants provide paired comparisons; each instance runs its
own case. These seeds differ from the previous staged-curriculum comparison.

Cases cover base3, ware2, jail1 and city1: Parasite+Infantry+Gunner trios
with MG/shotgun/blaster inventory and with the SSG inventory; also
Parasite+Infantry pairs with the SSG inventory on the second map sites.
The pool must close with all 144 captures usable and source unchanged before
comparison. Client, exporter, reward and native source fingerprints must
match, as must fixtures, loadouts, geometry and seeds.

`scripts/process_registered_combat_pool_cuda.py --plan-index 2` selects the
new model's 48 captures for CUDA replay verification. Selection does not
relax complete-pool acceptance or frozen-plan checks. Native first-life
outcomes determine goals, deaths, kills and health damage. Attack commands
do not establish hit accuracy. Lower total duration, including losing
fights, does not establish faster successful kills.

## Closed evaluation results

Pool closed with 144/144 usable captures and unchanged source. Paired
comparison receipts: `combat-gae95-vs99-quality-v1-20261010.json` and
`combat-parent-vs-gae99-quality-v1-20261010.json`.

| 48 identical validation fights | Parent | Lambda0.95 | Lambda0.99 |
|---|---:|---:|---:|
| Completed goals | 17 | 19 | 17 |
| Kills | 57 | 61 | 57 |
| Deaths | 27 | 27 | 29 |
| Dealt health damage | 11846 | 12207 | 12189 |
| Received health damage | 2822 | 3039 | 3299 |
| Game frames | 5282 | 5271 | 5026 |

| Composition/loadout, 16 fights each | Goals0.95 | Goals0.99 | Kills0.95 | Kills0.99 |
|---|---:|---:|---:|---:|
| Pair, SSG inventory | 11 | 9 | 23 | 20 |
| Trio, SSG inventory | 5 | 5 | 23 | 21 |
| Trio, MG/shotgun/blaster inventory | 3 | 3 | 15 | 16 |

Lambda0.99 gains three previously lost fights but loses five previously
won fights. Goals by map, lambda0.95→0.99: base3 4→4, city1 9→8,
jail1 0→0, ware2 6→5. No improvement is established; lambda0.99 is not
promoted. Lambda0.95's two extra wins over the parent on this small sample
also do not establish general superiority or justify live promotion.

Shorter aggregate duration includes deaths and is not a kill-efficiency
gain. Longer reward credit assignment alone does not solve the observed
group-combat failure. Further experiments should address useful discrete
decisions (target, attack, weapon) alongside continuous aim/movement, with
paired held-out checks; repeating lambda tuning on these seeds would be
validation-set fitting.

CUDA finalization of the selected 48 lambda0.99 captures completed:
`combat-gae99-quality-processing-v1-20261010/report.json`, 4863 eligible
rows, selected plan2, CUDA replay verification passed. The multi-model
plan selector is live-validated. No evaluation data was used for training.
