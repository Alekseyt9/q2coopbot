# H-JEPA utility-preserving predictor objective: rejected offline

Artifacts: `workspace/artifacts/hjepa-world-utility-v1-20261116`.
The protocol was recorded before fitting. Candidate and control used the same
TRAIN437 provider clips, proposed-action inputs, initialization, random minibatch
order, Adam1e-3 and 200 epochs. Encoders, utility readouts, actor, motor and weapon
head were frozen exactly. No validation checkpoint selection or weight sweep.

The candidate adds weight1 times the frozen utility error to latent prediction
MSE at all three horizons. Its target is the utility of the real encoded future
state. The control uses weight0. This preserves an existing readout; it does not
introduce native reward into policy inputs or prove that readout correct.

The zero-weight control reproduces the preceding proposed-input model exactly,
including all arrays. Independent Go parity compares1600 values per model with
maximum absolute errors2.22e-15 and3.55e-15. All frozen fields remain unchanged.

The candidate reduces TRAIN horizon16 utility error from0.0641 to0.0342 but
increases validation error from0.3452 to0.8517. Validation latent MSE at horizon16
also increases from0.1512 to0.1925, exceeding the allowed10 percent regression.
Thus improving the training objective did not generalize to these held-out clips.

The native-progress utility audit uses the same2416 overlapping VALIDATION440
clips as the preceding mechanism diagnostic. Exact metrics and the predeclared
offline gate are stored in `handoff.json`; all required criteria fail. The
candidate is rejected before native play. The generated VALIDATION442 plan was
checked to be disjoint from440/441 but was not executed or used for outcomes.

No model promotion, no U1 changes, no autonomous-combat completion claim.
Next investigate training coverage and direct native outcome increments with
separate source cohorts, rather than sweeping this coefficient on validation440.
