# Confirmed Blaster miss penalty: completed development evaluation

Both branches started with identical actor/std and reset critic/Adam, received
80 fresh own-policy fights each and one CUDA update with10 accepted actor steps.
The miss reward differs only by-0.02 per confirmed Blaster geometry/sky miss.
The evaluation completed480/480 native fights, zero execution errors, unchanged
source binding and complete member proof. All variants used the same original
evaluation reward, fixtures and guards. Final-test superiority is not claimed.

| Variant | Wins /80 | Deaths | Mean received health damage |
| --- | ---: | ---: | ---: |
| Control before | 48 | 23 | 35.4 |
| Control after | 52 | 19 | 33.9 |
| Miss penalty before | 48 | 23 | 35.9 |
| Miss penalty after | 52 | 23 | 38.0 |
| FireBC | 49 | 29 | 44.5 |
| Rules | 67 | 8 | 14.3 |

Both updates gained7 and lost3 wins relative to their own before runs. Between
the two after variants there were3 gained and3 lost wins, net0; separately,
Blaster40-fight and Machinegun40-fight strata both had net0. Miss after had
four more deaths and4.05 more mean received damage than control after. The
repeated identical parents had exactly the same win/death outcomes, while mean
received damage differed0.5375, showing some native repeat variation.

| Variant | Eligible native Blaster launches | Confirmed geometry/sky misses | Miss fraction | Unknown endings |
| --- | ---: | ---: | ---: | ---: |
| Control before | 410 | 149 | 36.34% | 0 |
| Control after | 422 | 169 | 40.05% | 0 |
| Miss before | 407 | 146 | 35.87% | 0 |
| Miss after | 450 | 157 | 34.89% | 0 |
| Rules | 204 | 5 | 2.45% | 0 |

The miss branch has a lower confirmed miss fraction than control after, but
only a small decrease from its own before run. It fired more eligible Blaster
shots, rather than obtaining the smaller fraction by reducing total launches.
These counts describe native mod1 projectiles across first-life eligible
windows; they are not Machinegun accuracy or selected-target hit proof.

Both after variants launched immediately in all40 initially-Blaster fights.
Their first sent attack delays were identical on average0.55 native frames,
maximum12. Initially-Machinegun ammo decrement proxy:40/40 observed, mean1.1,
maximum12frames, three over five frames. The penalty did not introduce a
measured first-shot hesitation in these cases; this is not a claim for every
weapon or untouched map.

Decision: **do not promote the miss objective on this result**. Wins are tied,
survival/received damage favor control, and both learned variants remain well
below rules. Keep the control reward for the next matched training iteration;
retain the optional miss recipe and all evidence for later reward experiments.
There is no automatic deployment of either checkpoint.

Next: integrate and validate first-life death stopping for subsequent corpora,
preserving the death reward and zero terminal bootstrap; use the verified CUDA
batch finalizer for fresh data. Continue work on aim/movement quality and longer
own-policy training rather than increasing the penalty solely on this sample.

Evidence under `workspace/artifacts/miss-reward-ab-eval-20261010`:
`acceptance.json`, `quality-report.json`, `miss-ab-comparison.json`,
`miss-reward-audit.json`, `first-attack.json`, `first-life-tail.json`, and
`recovery/verified-members.json`. All queued report helpers have exited.
