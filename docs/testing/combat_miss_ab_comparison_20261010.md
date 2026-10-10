# Paired miss-objective comparison

`scripts/report_combat_miss_ab.py` produces `miss-ab-comparison.json` and
`miss-ab-comparison.md` inside the evaluation root. It requires the complete
native member proof, common-reward quality report, episode rows, projectile
miss audit and first-attack diagnostics. Inputs are hashed and checked again
after calculation; partial cohorts are rejected.

It verifies all six variants have the same80 generated fights, seed lists,
loadouts and evaluation reward recipes. The two before variants must have
identical deterministic weight hashes. Episode keys, totals and member report
hashes must match the sealed proof and quality report. This is reporting only;
there is no neural inference, model training or reward rewrite.

Pairs, with positive deltas meaning the right variant minus the left:

- The two identical before actors: native repeat variation.
- Control before/after: improvement from the common training procedure.
- Miss before/after: improvement with the small confirmed Blaster miss cost.
- Control after versus miss after: direct candidate comparison.
- Rules versus miss after, and FireBC versus miss after: quality references.

Every pair reports gained/lost wins, both won/both lost, deaths and received/
outgoing damage changes, overall and separately by episode loadout. Difference
of update win deltas is descriptive: it does not establish a causal effect,
significance or superiority on an untouched final test. Training used equal
80-fight allocations, with different actual valid transition counts reported.

To detect a policy that simply avoids firing, variant diagnostics retain the
eligible Blaster launch count, confirmed miss count, unknown endings, creditable
miss count and first requested/sent attack delays. Initial-weapon launch/ammo
strata come from the first-attack report. Machinegun ammo decrement remains a
proxy; native Machinegun per-shot evidence is not yet available. Geometry/sky
miss accounting alone is not selected-target hit accuracy.

Validation completed so far: the pair accounting was run against the preceding
sealed480-fight corpus. It exactly reproduced Instant4 gained/8 lost/net-4
and Postmove9 gained/7 lost/net+2 from that corpus's quality report. Syntax and
whitespace checks passed. Full live acceptance of this new report remains
pending completion of the current miss-objective evaluation.

The report is queued behind the retained process handle of the first-attack
reporter, which itself waits for the evaluation driver. Its receipt is
`workspace/build/miss-ab-comparison-process.json`; queued errors are retained
in `workspace/build/miss-ab-comparison-v2.stderr.log`.

Manual command after the evaluation and diagnostics have completed:

```powershell
& F:/src/strat/.venv-gpu/Scripts/python.exe scripts/report_combat_miss_ab.py --root workspace/artifacts/miss-reward-ab-eval-20261010
```

At the current snapshot, the480-fight pool is live;82 closed job receipts exist
and none reports an execution error. This is capture progress, not a quality
result or promotion decision.
