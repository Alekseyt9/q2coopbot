# H-JEPA871 world/motor context contract

Added opt-in `feature_version="hjepa_context_features_v1"` to H-JEPA only.
It contains the unchanged845 combat-v6 prefix plus26 causal equipped/own previous
weapon-request values, identical in content to the existing separate weapon-head
context. World encoder input and direct motor input now follow this explicit
header. The optional weapon head keeps its independent845/871 input declaration;
its context is appended once. Existing845 files and PPO feature contracts remain
supported. No Python runtime inference service is introduced.

`q2hjepa-data --feature-version hjepa_context_features_v1` exports the new format
for both current and next states. Default export remains v6/845. Postmortem
inventory masking preserves observed equipment and does not modify native data.
Training consumers must explicitly admit the new version; the existing845-only
trainers do not silently reinterpret it.

Converter: `scripts/prepare_combat_hjepa_context_model.py`. It adds exactly26zero
columns to the first world encoder and optional motor, changes the explicit
feature header and verifies every other model field unchanged. These zero columns
do not yet use the new information or improve combat; they provide an auditable
starting point for representation and motor training.

## Validation

Artifacts: `workspace/artifacts/hjepa-context-world-v1-20261116`.

- `go test ./...` passed. Focused tests cover four planners and five weapon-head
  modes, exact zero-extension actions/latent predictions, stochastic likelihoods,
  held options, independently declared weapon widths, equipment/request influence
  through nonzero new weights, rejection of width mismatches, and terminal export.
- Independent Go parity accepted1600 values from original845 networks evaluated
  against the new871 model. Maximum error1.55e-15.
-32 native validation443 games,16 paired worlds, same frozen sampling seed261700.
  All outcomes and1623 first-life observation/action/applied-command pairs match
  exactly. Frozen cross-replay of both full captures accepts3260 proposals with
  zero mismatches.
- Both845 and871 exports of the same capture retain2065 transitions. Their
  prefixes and every non-feature row field match exactly.
- U1 SHA remains `21c3b680e0b6090d601d524f6da53d73ae6dd083c8e7a9a6da2e6d268ccadd26`.

The strict raw-capture-length condition did not pass: case7 has four additional
legacy provider rows at frames168..171 after verified goal endpoint164. These
rows are not first-life action credit. The models still exactly cross-replay both
complete streams, including those rows; no raw-length equality claim is made.
First-life compatibility and numerical agreement are separate from combat quality.

The matrix wrapper also exited after capture on job17's stream-compression
running-writer check. All32 native reports/proofs were already complete; retry
compression succeeded and completion was reconstructed with an explicit recovery
record. No native episode was rerun. The repeated postprocessing check warrants
separate investigation; its precise cause is not yet established.

All owned jobs finished. No training or live promotion occurred, old845 models and
U1 remain preserved. Next add typed871 offline learning support and learn world
representations from retained trajectories, with whole-world fit/holdout separation
and fresh external quality validation. Reserved external VALIDATION442 is unplayed.
