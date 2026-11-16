# Joint H-JEPA world representation fit

Artifacts: `workspace/artifacts/hjepa-joint-world-v1-20261116`.
Training uses the retained384 controlled TRAIN episodes on64 source worlds.
It does not clone diagnostic constant commands into a combat actor.

Go exported45378 typed871 transitions; source admission retained38001 provider
transitions and36753 continuous windows:27056 fit windows from48 worlds and9697
internal holdout windows from16 worlds. Source/frame/step identifiers are kept.
There are4525 shortened windows ending at verified terminal transitions. Their
next-state targets are absorbing and their known constant planned commands repeat,
but a true-length mask credits terminal effects exactly once. Nonterminal gaps,
control transfers and life/source changes cannot be padded. Ten focused tests
cover these boundaries, native cost accumulation and finite physical readouts.

Both845 and871 variants train their three encoders, predictors and native-effect
readouts from the common initial weights, with the same1000 sampled Adam steps.
The871 model starts with26zero input columns. EMA0.99 supplies latent targets;
variance/covariance regularization counters collapse. Effects use sigmoid
progress/terminal outputs and nonnegative softplus damage costs. Only native
effects supervise readouts; historical accumulated-progress utilities are not
assumed calibrated after encoder changes. Actor, motor and weapon policy are
not optimized. The fixed final step is used without selecting step500.

## Internal holdout

| Horizon |845 reward MSE |871 reward MSE | Fit-mean reward MSE |871 prediction/persistence MSE ratio |
|---|---:|---:|---:|---:|
| 1 |0.008484 |0.008504 |0.008910 |1.030 |
| 4 |0.029103 |0.028436 |0.034644 |0.899 |
| 16 |0.080890 |0.077773 |0.126439 |0.795 |

The871 model beats the mean forecast at horizon16, but its gain over the845
variant is only3.9 percent, below the predefined10 percent. Horizon1 fails to
beat persistence. Overall offline gate FAILED. The new context columns have
L2 norm1.344, proving they were trained; this does not establish better control.

The heldout initial-state controlled contrasts expose a further limitation.
At horizon16,871 predicts the correct sign in35/58 nonzero comparisons, but its
contrast RMS is only0.00577 versus actual0.21309 (2.7 percent of actual scale).
Contrast MSE0.045148 barely improves zero-contrast0.045406. It predicts much of
state-dependent risk while underrepresenting the effect of choosing another
command. Comparisons share16 worlds, not80 independent trials. These internal
worlds were already diagnostic and are not fresh external quality evidence.

## Runtime status and next stage

Both outputs deliberately use `hjepa_joint_world_offline_v1`; Go rejects their
format. Existing utility scoring cannot be attached to changed encoders without
an explicit compatible objective. No native quality evaluation, model promotion
or U1 replacement occurred. Reserved external VALIDATION442 remains unplayed.
All owned preprocessing/training jobs finished. The frozen trainer and digests
preserve the actual fit; a subsequent scalar-reporting detach correction only
removes a PyTorch warning and does not require retraining these models.

Next prioritize a learned action policy connected to the representation. Start
with an exact deterministic bridge from the existing871->128->64->8 motor to a
64-dimensional encoder and action head, preserving the existing proposal before
adding stochastic full-primitive on-policy learning. World prediction can then
serve as an auxiliary objective while real native outcomes train combat actions.
This continues the autonomous-combat goal; it does not promote the rejected risk
readouts or treat constant diagnostic controls as expert behavior. Preserve old
models, source checkpoints and U1 throughout.
