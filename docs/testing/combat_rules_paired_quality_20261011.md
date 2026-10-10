# Common-source combat comparison, 2026-10-11

All 240 captures completed with unchanged source and usable manifests: ordinary
rules, frozen decision update9, and continued decision update10, 80 test battles
each. The 16-slot pool used timescale2 and matched recipe/fixture/seeds across
controllers. Paired checks bind client/exporter/native source, actual runtime
file hashes and reset controls. This does not prove equivalence of every
internal AI/world state.

| Metric | Rules | Update9 | Update10 |
|---|---:|---:|---:|
| Goals | 26 | 29 | 32 |
| Kills | 116 | 106 | 111 |
| Deaths | 46 | 42 | 41 |
| Damage dealt | 23600 | 22229 | 22790 |
| Damage received | 4990 | 5454 | 5591 |
| Recorded frames | 8761 | 9579 | 10431 |

| Map,20 battles each | Rules goals | Update9 goals | Update10 goals |
|---|---:|---:|---:|
| base3 | 6 | 5 | 8 |
| city1 | 10 | 13 | 12 |
| jail1 | 3 | 0 | 0 |
| ware2 | 7 | 11 | 12 |

The increased goal count is accompanied by fewer kills, more received damage
and more recorded frames. Jail1 remains a major regression. Update10 is not
promoted; update9 remains the experimental training reference. These 80 matched
cases do not establish overall superiority, optimal actions or campaign/live
coop acceptance. The evaluation cases must not become training data.

Authoritative artifacts under `workspace/artifacts/`:

- `combat-rules-ab-pool-v1-20261010/report.json`;
- `combat-rules-vs-update9-quality-v1-20261010.json`;
- `combat-rules-vs-update10-quality-v1-20261010.json`;
- `combat-rules-ab-update9-processing-v1-20261010/report.json`;
- `combat-rules-ab-update10-processing-v1-20261010/report.json`.

Both learned streams completed CUDA verification: update9,9371 eligible rows;
update10,10111 eligible rows. No training was performed during finalization.
The rules stream is excluded from PPO finalization by explicit plan selection;
whole-pool capture acceptance still includes every rules and learned case.

Eight native-tested second-site trio recipes are now registered in the global
index: base3,ware2,jail1,city1, each with ordinary weapons or the SSG inventory.
Their bytes match the frozen common-source comparison draft and allow both
rules and learned modes. Registration receipt:
`registered-expanded-sites-v1-20261011.json`; validated CLI listing:
`registered-expanded-sites-list-v1-20261011.txt`. Previously frozen private
registries and evaluation plans remain unchanged.

Next implementation: shared per-enemy target representation, motivated by the
feature-use diagnostic, with zero residual migration followed by fresh
own-policy captures and held-out paired evaluation.
