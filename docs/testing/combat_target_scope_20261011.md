# Target-only fork experiment, 2026-10-11

The previous common-source80-case benchmark regressed from33 parent goals to30
for decision control and28 for shared decisions. Both trained models lost
ordinary-weapon trio goals11→6. This suggests testing a narrower update; it
does not prove attack or weapon learning caused the regression.

Two explicit forks resume the same original checkpoints used in the previous
training experiment, not the regressed update10 checkpoints. Each reuses its
own accepted64-case train-only corpus and identical configuration/optimizer
history,10 accepted actor steps, target-row learning rate0.003. No held-out
frames or other controller's rollout is used for PPO.

- Control target-only:8369 eligible rows,1170 trainable actor scalars.
- Shared target-only:8376 eligible rows,17363 trainable actor scalars, including
  the16193-parameter per-enemy shared residual.

Both CUDA checkpoint audits passed.111926 inactive actor scalars, log_std and
inactive Adam moments remain exactly unchanged, including attack, weapon,
movement, vertical, aim and aim-mode outputs and encoder/spatial parameters.
462 selected scalars changed in control;9472 in shared. Each fork appends only
its train corpus SHA to consumed rollouts and increments update9→10. The critic
trains normally. Parameter preservation does not imply identical trajectories
after the selected target changes.

The audit now supports both declared target and decision scopes. Rechecking
the previous decision control also passed, preserving110236 inactive scalars.

Fresh held-out comparison:20 recipes×4 test seeds at offsets12:16×6 controllers,
480 battles on the16-slot episode queue, timescale2. Controllers are frozen
parent, previous decisions control, previous shared decisions, target-only
control, target-only shared, and rules. All controllers use matching fixtures
and seeds. This permits direct target-only versus decisions comparison for
each architecture on the same new sample. Captures must close with unchanged
source; native paired comparisons and CUDA replay remain required.

Current weights are not promoted. Published models and live coop remain
unchanged. No new train capture or duplicate full training export was created
for this experiment. Closed exported streams are compressed after producer
exit with unchanged logical SHA checks; monitor physical disk space.

Artifacts: `combat-target-only-{control,shared}-update-v1-20261011`, their CUDA
checkpoint audits, and `combat-target-scope-eval-*-v1-20261011` under
`workspace/artifacts/`.

Post-capture analysis is queued in
`combat-target-scope-evaluation-analysis-v1-20261011/progress.json`. Its
coordinator binds the live capture PID and creation time and requires the
complete480-case pool with unchanged source before proceeding. It then runs
five paired native comparisons against the parent, a common-baseline strata
summary, five80-case CUDA finalizers (two concurrently), and five native
behavior audits with a matched-fixture summary. It produces `complete.json`
only after all stages succeed; failure leaves an explicit failed progress
record. This is one finite experiment, not a recurring job. No promotion is
performed by the coordinator.
