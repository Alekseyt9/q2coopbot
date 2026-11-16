# H-JEPA shared actor and first primitive PPO pilot

Experimental branch; the existing U1 runtime remains unchanged. Artifacts live
under `workspace/artifacts/hjepa-actor-bridge-v1-20261116` and
`workspace/artifacts/hjepa-actor-ppo-v1-20261116`. This is a structured-state
H-JEPA adaptation, not a replication of the paper's visual system.

## Bridge and native sampling

The old motor was 871 → 128 → 64 → 8. Its first two layers now form
encoder0, followed by an identity 64 layer; the final motor layer becomes
the actor. This preserves outputs exactly while exposing a shared trainable
representation. Planner `bc` executes actor proposals. Newly initialized
upper world encoders/predictors and zero utilities currently do not choose
actions and are not used as auxiliary gradient targets in this first PPO pilot.

Native TRAIN510 comparison: 16 worlds × four variants = 64 games.
Original motor and new actor agree on all 2345 first-life observation,
proposal and applied-action pairs, with equal native outcomes. Two raw
capture lengths differ after the goal endpoint; those tails are not parity
failures inside the first life. Go bridge parity: 2112 values, maximum
absolute error 1.33e-15.

Four stochastic primitive streams use counter seeds 261980–261983 on the
same 16 TRAIN worlds. Movement raw standard deviation is 0.15, aim 0.003,
fire/pose temperature 0.25. Frozen weapon selection uses seed 261700.
Streams 0/1 came from the initial matrix; streams 2/3 from a second completed
32-game matrix. They are 64 actual sampled episodes on 16 distinct worlds.

Go exports preserve raw Gaussian latents, Bernoulli fire, categorical pose,
joint weapon likelihood, source hashes and native dispatch receipts. The
Gaussian density is evaluated before tanh; fixed Jacobians cancel in PPO
ratios. Proposed actions define likelihood, with applied guards treated as
part of the environment. This is not the existing PPO-provider data format.

Independent Python float64 audits pass for all 8269 proposals:
2147 / 2045 / 2310 / 1767 by stream. Maximum joint log-probability error is
1.89e-13. Fire remains almost always active under the inherited confident
logits; this exploration setting has not demonstrated good combat behavior.

## First update

`scripts/train_combat_hjepa_actor_ppo.py` imports JSON directly into float64
CUDA layers. It admits only source-bound matched provider decisions, rejects
recovery commands, verifies independent stream weights and paired initial
observations, and masks nonexclusive or post-first-life/terminal decisions.
8245 decisions are admitted; 24 are masked.

Credit is canonical native first-life return-to-go, with a time-aligned
leave-one-other-stream-out baseline and scale 100. Reward is:

`100 win − 100 death + 10 capped monster HP progress − 0.1 received HP − self HP − 0.5 teammate HP`.

Native effect joins, actual health loss, terminal receipts and scalar-return
balance are checked by `hjepa_weapon_returns.py`. Outcomes and raw behavior
logits are labels/proofs only; neural inputs are the current 871 features.

Four full-batch Adam updates use learning rate 1e-4, clipped PPO ratio 0.1,
gradient norm limit 1 and mean behavior KL limit 0.02. A predefined line
search scales each proposed parameter delta by powers of two through 1/4096;
acceptance requires finite values, KL within the limit and no surrogate
decrease. Rejection restores both model and optimizer. No critic is claimed.
Only encoder0, actor and four Gaussian log standard deviations are updated.
Weapon and auxiliary world heads stay byte-value identical.

Accepted parameter scales: 1/64, 1/32, 1/16, 1/32. Final mean KL 0.00235364;
maximum per-observation KL 0.0232279 (the configured bound is on the mean).
Initial Go/CUDA likelihood discrepancy is 1.41e-13. This establishes a
connected, bounded policy update, not better play.

True 2+2 versus four continuous CUDA updates match model tensors, Adam
moments, history, input bindings and exported model JSON bytes exactly.
Updated Go actor parity passes 176 values with maximum error 2.67e-15.
Five distribution/precision/terminal-mask tests and nine native-return tests
pass. All Go packages passed after the runtime/exporter changes, before
these Python-only trainer/report additions.

## Fresh screen

Before gameplay, `evaluation-protocol.json` fixed validation offset444:
16 fresh generated worlds across eight known maps, six policies, 96 games.
Compare parent and candidate with primitive means and with one coupled
primitive stream, plus frozen live-U1 weights and rules. Primitive-mean
evaluation still uses the frozen stochastic weapon head. Each variant has
16 actual games; streams do not create additional independent worlds.

The pilot screen requires at least two additional wins with no increase
in deaths, received HP damage or self damage versus its parent in both
modes. Results belong in `native-screen.json` after matrix completion;
`scripts/report_combat_hjepa_actor_ppo.py` verifies the paired model changes
and reports map-cluster intervals. Reserved validation442 remains unplayed.
There is no promotion and no demonstrated superiority over rules yet.

### Completed validation444 result

| Variant | Wins | Deaths | Timeouts | Received HP | Self HP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parent, primitive means | 3 | 10 | 3 | 1204 | 217 |
| PPO, primitive means | 3 | 10 | 3 | 1200 | 185 |
| Parent, sampled primitives | 4 | 8 | 4 | 1151 | 110 |
| PPO, sampled primitives | 4 | 8 | 4 | 1126 | 100 |
| Frozen U1 | 5 | 7 | 4 | 997 | 106 |
| Rules | 6 | 6 | 4 | 805 | 70 |

The pilot gate **failed**: no win gain in either mode. Small damage/return
changes are descriptive findings on this cohort, not established improvement.
Do not fit on these validation worlds or promote the candidate.

The original matrix completed all 80 learned-policy games, then all 16 rules
jobs failed preflight before gameplay: the launch manifest incorrectly gave
rules the learned-only `stock_weapons_fire_v1` contract. Preserve the original
stderr and manifests. `native/capture-recovery.json` binds the 80 verified
captures and all 16 preflight errors; `native/complete.json` explicitly
describes that recovered five-policy subset. The 16 rules games completed
separately under `rules-native` with supported `legacy_v1` dispatch. This
reference includes guard/dispatch differences; parent/candidate comparisons
use identical contracts. `native-screen.json` verifies both identical fixture
cohorts and records the distinction.

U1 weights still hash to
`21c3b680e0b6090d601d524f6da53d73ae6dd083c8e7a9a6da2e6d268ccadd26`.
No Go or PowerShell source changed during either capture. Final stricter
admission checks reproduce the fitted weights and report exactly
(`final-admission-recheck.json`). Trainer/test/reward/report sources are
snapshotted under `source`; final report source is updated after the separate
rules cohort support was added.
