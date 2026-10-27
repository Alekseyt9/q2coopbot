# Five-frame movement-option training: 2026-10-27

## Scope and implementation

The existing v13 temporal-attention policy remains responsible for aim, firing,
weapons and the underlying Gaussian movement commands. A new categorical head
chooses one of five movement mean shifts (none, backward, forward, left, right)
and holds it for five provider frames. Conditional aiming and firing still see
the actual sampled movement. No scripted retreat or combat teacher is applied.

The option actor and option value networks each use the current 1554 features
and the frozen causal actor's 81 outputs. Sizes are 1635→32→16→5 and
1635→32→16→1 (52,965 and 52,897 parameters). Only these new networks are fitted.
The source is update-09 from `combat-stock-full-series-v3-20261026`.

## Closed evidence

- Native smoke: 6/6 game captures and 6/6 CUDA likelihood checks passed.
  Disabling options preserved original native behavior. Enabling them changed
  movement and held each selected choice for five frames.
- Synthetic optimizer test: eight steps and four steps/save/reload/four steps
  produced exactly equal parameters and Adam states. This was not gameplay fit.
- Fresh TRAIN: 53 registered cases across 38 campaign maps, seed offset 400.
  Existing independent episode queue, 16 slots, timescale 2. Native collection
  took 221.292 seconds; all 53 native captures and all 53 CUDA replays passed.
  Random native monster groups and partial stock inventories came from the
  existing registry. These seeds were not in the source checkpoint's 477
  consumed TRAIN seeds.
- First actual CUDA update: 7,258 native transitions, 7,250 loss transitions,
  1,468 movement decisions, eight actor and eight value steps. Approximate KL
  0.0006108434; recorded initial option log-probability/value matched exactly.
  The exported checkpoint reopened successfully with matching Adam states.
- All original policy parameters remain unchanged. No promotion occurred.
- Independent read-only CUDA audit reproduced the first fit's KL and confirmed
  the frozen source dictionary is identical. Training PPO objective changed
  from -0.0007776640 to -0.0032140855; option-value MSE from 10.3164005 to
  10.2336407. Mean probabilities on TRAIN decision states changed from
  [0.8, 0.05, 0.05, 0.05, 0.05] to
  [0.7891821, 0.0496096, 0.0507076, 0.0552374, 0.0552633]. These describe the
  update on its training batch, not validation quality.

The untrained-option TRAIN collection had 15 goals and 30 deaths in 53 cases.
These are collection outcomes, not a before/after validation result.

## Return and boundary contract

Categorical PPO loss is computed once per option decision, not on held frames.
Rewards are summed with per-frame discount; bootstrap discount is gamma raised
to the actual duration. Lambda applies once per decision. Terminal short options
retain every observed reward and use the verified terminal bootstrap.

For a nonterminal partial final option, the loss horizon moves back to that
option's start. The preceding complete option bootstraps from the recorded value
at that decision boundary. The incomplete tail is explicitly reported and never
treated as death or victory. All causal history is retained. First update
censored eight frames across four cases (2, 1, 4 and 1); all 53 cases contributed.
Missing histories, partial loss starts and unsupported resets fail admission.

The next fresh TRAIN batch exposed two complete five-frame options whose next
decision was never called after combat ownership ended. This originally failed
admission before any parameter update. It is now handled by evaluating the old
option critic on the observed `next_features` with the frozen actor's causal
history extended by that observation. No future action or synthetic reward is
introduced. A GPU test masks an actually recorded next decision and reproduces
its nonzero value; the observed provider prefix remains identical. Complete
options are retained, while unsupported partial options still cannot use a
free-decision bootstrap. Evidence:
`combat-movement-option-bootstrap-check-v1-20261027/report.json`.

Update 2 subsequently completed on the already collected 53 fresh TRAIN cases
(offset 401), resuming both Adam states from update 1. It used 7,931 native rows,
7,918 loss rows and 1,598 decisions; thirteen partial-tail frames were explicitly
censored and both complete missing-successor windows were retained. Eight actor
and eight value steps passed, approximate KL 0.003057888. Checkpoint reopened,
original source stayed frozen, and cumulative consumed TRAIN cases reached 106.
Evidence: `combat-movement-options-fit-v2-20261027/report.json`. This is an
implementation/training milestone; update 2 has no separate quality result.

## Artifacts

- `workspace/artifacts/combat-movement-options-native-smoke-v3-20261027/analysis/report.json`
- `workspace/artifacts/combat-movement-option-resume-v1-20261027/report.json`
- `workspace/artifacts/combat-movement-options-train-v2-20261027/complete.json`
- `workspace/artifacts/combat-movement-options-fit-v1-20261027/report.json`
- `workspace/artifacts/combat-movement-options-development-v1-20261027/returns-check.json`
- `workspace/artifacts/combat-movement-option-fit-audit-v1-20261027/report.json`

The fit used the preserved development trainer and return helper in that
development artifact directory. Their production copies are
`scripts/train_combat_movement_options_cuda.py` and
`scripts/combat_movement_option_returns.py` (imports/root discovery adapted).

Four-arm validation is recorded separately under
`combat-movement-options-eval-v3-20261027`: source U9, initial options, fitted
options and rules, 38 shared validation worlds per arm, seed offset 401.
All 152 native captures and 114 CUDA checks completed successfully; independent
closure audit is `combat-movement-options-eval-audit-v1-20261027/report.json`.

| Arm | Goals / 38 | Deaths | Kills | Monster HP dealt | Received HP | Self HP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen source U9 | 7 | 27 | 43 | 6786 | 2778 | 309 |
| Initial options | 9 | 25 | 41 | 7647 | 2498 | 284 |
| Fitted options update 1 | 10 | 24 | 42 | 7744 | 2575 | 232 |
| Rules | 11 | 17 | 50 | 8609 | 2085 | 0 |

Fitted vs source gained jail4, mine3 and hangar1 without losing any source win.
Fitted vs initial gained mine3 and hangar1 but lost ware1: only one net win.
Rules retain higher goal count and fewer deaths. This small descriptive result
does not establish learned superiority; no promotion occurred. These worlds
differ from the earlier 14/38 reference batch, so its absolute totals are not a
before/after baseline for this table.

Read-only damage attribution on this closed validation set is recorded in
`combat-movement-options-damage-audit-v1-20261027/report.json`. Its per-event
self damage reconciles exactly with the first-life native summaries. For
update 1: five cases incurred 232 self HP and all five ended in death, but only
three deaths were directly caused by a self event (jail5 grenade splash,
power1 rocket splash, boss1 BFG blast). The other two are associations, not
proven causes. Self damage comprises grenade splash 110 HP, rocket splash
29 HP, hand-grenade splash 32 HP and BFG blast 61 HP. Damage codes were checked
against the exact native engine source header hash.

Received damage is inclusive of self damage. The 490 HP difference from rules
(2575 versus 2085) decomposes as: self +232, worldspawn +78, target_laser +71,
known monster classes +113, and unresolved `freed` attacker -4. These are
descriptive attributions across different trajectories, not a counterfactual
claim that avoiding self damage would recover three wins. Validation observations
are not training labels. Current learning config remains unchanged.

Update 3 also closed on fresh TRAIN offset 402: 53 native/CUDA cases, 8,094
native rows, 8,088 loss rows and 1,630 decisions, six partial frames censored,
two missing boundary values reconstructed with unchanged causal prefixes,
eight actor/eight value steps, approximate KL 0.000313238. Both optimizer states
and the frozen source were verified; cumulative TRAIN cases reached 159.
Evidence: `combat-movement-options-series-v1-20261027/update-03/fit/report.json`.
No separate quality result exists for update 3.

## Five-update result

The fixed series closed at update 5 with 265 fresh TRAIN cases. The independent
checkpoint chain audit `combat-movement-options-chain-audit-v1-20261027/report-update05.json`
verified no repeated TRAIN seeds, no overlap with the source checkpoint's
registered TRAIN seeds, identical frozen source policy, and continuous Adam
clocks (40 steps for both option actor and value). Update 5 used 7,247 native
rows, 7,232 loss rows, 1,459 decisions, fifteen partial frames censored and two
observed-next-state boundary reconstructions. Eight actor/eight value steps;
approximate KL 0.006267608. Weights SHA256:
`28b97a4fef6ff478179b236bb64469dbc8ce2d556f064ecc1d167afa70b8e643`.

The cached-control evaluation added 38 new native games and 38 CUDA checks,
reusing the 152 sealed source/initial/update1/rules games on identical generated
validation conditions. Independent closure:
`combat-movement-options-update05-audit-v1-20261027/report.json`.

| Variant | Goals / 38 | Deaths | Kills | Received HP | Self HP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source U9 | 7 | 27 | 43 | 2778 | 309 |
| Initial options | 9 | 25 | 41 | 2498 | 284 |
| Update 1 | 10 | 24 | 42 | 2575 | 232 |
| Update 5 | 8 | 25 | 38 | 2538 | 203 |
| Rules | 11 | 17 | 50 | 2085 | 0 |

Update 5 lost mine3 and hangar1 relative to update 1 and gained no wins. It did
not improve the initial untrained options. Continuing movement-only training
did not establish learned superiority; no weights or defaults were promoted.

Update 4 initially stopped before fitting because one TRAIN case had a real
ownership gap: provider frames 98–274, rules 275–339, provider again from 340.
Return construction now splits contiguous admitted ownership segments, censors
incomplete option edges and disallows categorical loss for held prefixes whose
decision was not admitted. Complete-boundary reconstruction also supports each
segment end, rather than only the final row of a seed. GPU test
`combat-movement-option-segments-check-v1-20261027/report.json` verified real
sampling/context and exact independence of earlier returns when rewards after
the gap were altered by 1000. No rules transitions entered option PPO.

Recovered update 4 is sealed under `combat-movement-options-recovery-v1-20261027/fit-04`:
53 native/CUDA cases, 8,828 native rows, 8,823 loss rows, 1,782 decisions, five
partial frames censored, eight actor/eight value steps, KL 0.002912117. Both Adam
states and the frozen source were verified; cumulative TRAIN cases are 212.
The earlier failed series and evaluator receipts remain historical terminal
attempts. The active replacement series is `combat-movement-options-series-v2-20261027`;
the replacement queued comparison is `combat-movement-options-update05-eval-v2-20261027`.

A separate TRAIN-only audit of all four sealed collections is
`combat-self-damage-train-context-v1-20261027/report.json`: 212 cases, 36 with
self damage, 51 self-damage events, 2,027 self HP, 12 directly lethal self events.
At one frame before impact, all 51 observed contexts exist; the future native
inflictor ID matches a visible candidate projectile in 20 cases (all in the
nearest four), which does not prove projectile lifetime identity. Four
`MOD_HELD_GRENADE` events total 200 HP. In all four, observations at leads
1/3/5/10 show the hand-grenade viewmodel, gun frame 11, requested and applied
attack true. Native weapon code confirms prolonged holding can detonate a
grenade in hand. This supports testing fire-cycle learning separately; it is
not a preferred-action label, and avoiding all resulting deaths is unproven.
No neural execution, new games or fitting occurred in this diagnostic audit.

Initial collection v1 and validation v1/v2 stopped during metadata planning,
before game collection. They are not combat losses. Hindsight at localhost:9077
was unavailable during this work; the verified findings remain here for later
ingestion. It was subsequently restored by starting its existing PostgreSQL
data directory and API without changing models or enabling MemOS. The closed
fit and validation findings were submitted to Hindsight with correction of its
older page's "no actor fit" status; queue acceptance is not proof of completed
knowledge-page processing.
