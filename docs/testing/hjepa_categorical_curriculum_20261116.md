# H-JEPA categorical exploration and sequential actor learning

Artifact root:
`workspace/artifacts/hjepa-categorical-curriculum-v1-20261116`.
This continues the failed optimizer screen in
`hjepa_natural_actor_20261116.md`. Existing U1 is preserved; no promotion.

## Fixed design

Original actor-bridge weights are the source, not the rejected natural or
Adam candidates. Raise the existing global categorical temperature from
0.25 to 1. This changes both fire/release and the release/jump/crouch pose
distribution, so it is not a fire-only intervention. Gaussian exploration,
weapon head and weapon sampling seed are unchanged at initialization.
There is no grenade-release rule, expert-action label or diagnostic override.

Three sequential on-policy batches use TRAIN508, 507 and 506: 16 distinct
worlds and four primitive streams each, 192 actual games on 48 worlds.
After each batch, four full-batch natural-gradient clipped-surrogate updates
train encoder0, actor and four log standard deviations. Carry fitted policy
weights into the next batch. This solver has no persistent optimizer moments.
Weapon and auxiliary world heads stay frozen. No old batch is reused for
another batch's gradient, and no evaluated validation data is admitted.

Plans and the comparison protocol were written before gameplay. Their seeds
do not overlap 75 previous H-JEPA plan files. This check covers local H-JEPA
artifacts, not unknown external experiments. All 48 TRAIN worlds and the
16 validation446 worlds are pairwise disjoint. Reserved442 remains unplayed.

`scripts/run_combat_hjepa_actor_curriculum.py` freezes its source dependencies,
parent and plan hashes; completed stages can be reused. An incomplete native
or fit directory requires process inspection instead of an automatic restart.
All native matrices use at most 16 concurrent games at 2× speed. Go/PowerShell
source must stay unchanged during capture. Python code snapshots are retained
under `source` and checked before every pipeline command.

Each batch requires exact Go source replay, independent float64 likelihood
audits and native first-life causal admission. Outcomes provide reward labels
only. Keep the preceding reward, other-stream time-aligned return baseline,
PPO clipping and mean-KL gate unchanged. After every fit, check updated actor
parity in Go and equality of frozen components. The first batch also performs
an actual 2+2 versus four CUDA-update resume check.

## Learning versus added randomness

The fresh validation446 screen contains 160 games on 16 worlds:

- Original cold policy, two coupled primitive streams: 32 games.
- Untrained temperature1 policy, the same two streams: 32 games.
- Final trained temperature1 policy, the same two streams: 32 games.
- Source and final primitive means, frozen U1 and rules: 16 games each.

Primitive-mean variants retain the frozen stochastic weapon head. All sampled
variants use primitive seeds262500/262501 and weapon seed261700. Learned
variants share `stock_weapons_fire_v1`; rules uses its supported `legacy_v1`
contract, so that controller reference includes dispatch/guard differences.

The learning gate compares trained to untrained hot behavior: at least four
additional wins over 32 games with no increase in deaths, received HP or self
HP. Primitive means separately require at least two additional wins over the
source's 16 games and the same nonregression checks. Cold versus hot measures
the distribution change without learning. No result from this screen alone
establishes unseen-map transfer, independent replication or promotion.

## Read-only fire probe

`audit-fire-probe.py` evaluates every completed model on the same 7906 prior
TRAIN509 observations: 537 observed hand-grenade gunframe11 states and 7369
other states. It never feeds labels into this curriculum. All probe fire
probabilities use temperature1 to separate fitted weights from the mechanical
temperature change; primitive-mean release counts use raw fire logit below0.
The source release probability on hold states is 5.1898% and on other states
6.1305%. These are counterfactual forward passes, not native combat outcomes.

Round1 completes 64 native games, exports7428 proposals and admits7389.
All four likelihood audits and updated Go parity pass. Its true 2+2 resume
matches weights, histories, bindings and exported JSON bytes. Native provider
hold decisions:285, proposed/applied releases:14, held-grenade events:5,
capped held-grenade health loss:197 HP. These are fresh TRAIN observations,
not a paired quality comparison to the older dataset.

After round1, fixed-probe hold release probability is 5.2186%; other-state
release probability is6.1596%. Primitive means still release on zero of the
537 fixed hold observations. This small weight effect is not learned fire-cycle
mastery; subsequent predefined batches and fresh native evaluation remain
necessary.

Four focused tests check first-life/provider/exclusive counting, capped native
health loss, missing-damage rejection and immutable prepared inputs. Tests pass;
the frozen report implementation emits non-fatal file-handle ResourceWarnings
under unittest's warning filter. Close those readers after the running source
freeze ends; do not change frozen pipeline sources during capture/fitting.

Authoritative progress is `state.json` and `training-progress.json`; a process
handle or actual process state must establish liveness. Completion/results
belong in `complete.json` and `native-screen.json` after the full pipeline ends.

## Completed result: no promotion

All three batches and both validation matrices completed normally:352 actual
games,192TRAIN and160validation. Export22187 proposals, admit22091, mask96.
All12 source likelihood audits pass; each round's Go parity passes176 values.
The first-round true2+2 resume passes exactly. The final model SHA256 is
`c8849bc543563108c9452e922e302e6eb48808006069663250377f2aa41ec53d`.

| Policy | Games | Wins | Deaths | Timeouts | Received HP | Self HP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold, untrained | 32 | 8 | 16 | 8 | 1627 | 96 |
| Hot, untrained | 32 | 8 | 14 | 10 | 1640 | 63 |
| Hot, trained | 32 | 7 | 17 | 8 | 1890 | 218 |
| Source primitive means | 16 | 4 | 6 | 6 | 833 | 134 |
| Trained primitive means | 16 | 5 | 8 | 3 | 819 | 93 |
| Frozen U1 | 16 | 4 | 6 | 6 | 702 | 46 |
| Rules | 16 | 5 | 4 | 7 | 504 | 0 |

The predefined sampled and mean learning gates both fail. More random actions
alone have the same win count with fewer deaths on this cohort; the trained
candidate regresses versus that matched untrained source. The extra mean-mode
win accompanies extra deaths. Neither establishes useful learning or justifies
promotion. Validation446 is now viewed and must not become training data.

Native TRAIN provider hold decisions/releases by batch:285/14,391/21,368/21.
Thus56 actual release examples occur across the three fresh batches. Native
held-grenade HP losses are197,186,199 on those different world sets; they are
not paired quality comparisons between successive policies.

Final fixed-probe release probabilities at temperature1 are5.1222% on hold11
and6.0592% on other states, versus5.1898%/6.1305% initially. All537 fixed hold
states still propose attack under primitive means. Validation trained sampled
policies release3times on58provider hold decisions, versus0/34 for untrained
hot; both have one held-grenade explosion. Different survival, weapon use and
state exposure prevent treating this conditional-count difference as learning.
The unchanged-probe probabilities argue against a learned release improvement.

## Numerical diagnosis, separate from the frozen experiment

`scripts/audit_combat_hjepa_actor_credit.py` performs a read-only initial-score
audit on admitted round1 TRAIN data. Gaussian gradient norm48.0824, fire0.04984,
pose0.23762. Hold11 fire contribution norm0.01861, other0.03729; cosine+0.538.
Factor gradients sum to the full score within1.87e-14. These score norms use
different action units; their ratio alone does not prove an incorrect loss or
cause of failed combat. The observed hold/other gradients are not conflicting.

The existing40-step CG solves do not meet the specified1e-10 residual target;
round2 update1 even has relative residual1.893. Nonlinear surrogate and mean-KL
acceptance still pass, so this is a solver-accuracy limitation, not evidence
that the accepted mean-KL bound was violated.

New isolated `hjepa_fisher_diagonal.py` computes the exact local Fisher diagonal
for the current Linear/ReLU primitive network, retaining the full pose covariance,
and supplies a Jacobi-preconditioned CG prototype. It is not imported by the
completed trainer. Four tests compare every small-network parameter against
independent full autograd Hessian diagonals at temperatures0.25/1, compare PCG
to a dense correlated SPD solve across six orders of coordinate scale, and
check invalid/zero cases. All pass.

On the same initial TRAIN system with40iterations and unchanged weights:

| Direction diagnostic | CG | Jacobi-PCG |
| --- | ---: | ---: |
| Verified relative residual | 0.1461 | 0.5091 |
| Damped quadratic objective | 0.10717 | 0.31070 |
| Predicted gain at damped KL0.02 | 0.09259 | 0.15766 |
| Scaled final fire-row parameter norm | 0.000248 | 0.039505 |

PCG improves the local quadratic objective and changes the action-factor
directions, but its raw residual is worse and neither solve is converged.
These are parameter-direction diagnostics, not fitted candidates, accepted
nonlinear updates or native quality gains. A source-matched PCG training pilot
on known TRAIN data followed by fresh validation is the next test; do not
claim that preconditioning has solved the combat or fire-cycle problem.

After the original frozen pipeline ended, report JSONL readers were changed
to context-managed UTF-8 streams. The four report tests pass with ResourceWarning
treated as an error. The original `source` snapshot and its manifest remain
unchanged as evidence of the actual run; `report-reader-fix.json` records the
later nonnumerical source repair. No native result or fitted model was rewritten.

Live U1 still hashes to
`21c3b680e0b6090d601d524f6da53d73ae6dd083c8e7a9a6da2e6d268ccadd26`.
No Go or PowerShell source changed during any capture in this phase.
