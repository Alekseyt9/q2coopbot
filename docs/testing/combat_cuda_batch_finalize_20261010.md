# CUDA batch finalization

`scripts/finalize_combat_cuda_batch.py` runs the existing, unchanged
`finalize_combat_cuda_rollout.finalize` function for independent native corpora
inside one interpreter. PyTorch imports and CUDA runtime startup are shared;
each case reconstructs its model, tensors and sequence context. Nothing carries
recurrent state or attention context across fights. Numerical verification and
bootstrap remain on CUDA, with no CPU neural fallback.

The request pins every model and pending native report. Fresh output paths are
required; the batch receipt stores each ready report and CUDA verification
hash. Each ready report additionally pins the batch request and entrypoint.
Failure preserves partial outputs with a failed receipt and does not mark the
batch complete. Sealed outputs cannot be overwritten. A partial failed batch
currently needs new output paths; no automatic in-place resume is claimed.

The processor exposes `--cuda-batch-finalize`, requiring `--cuda-only-export`.
The option is pinned in the processing protocol as
`independent_cases_single_interpreter_v1`; numerical/source snapshots remain
frozen. Existing non-batch protocols and completed training runs are retained.
The current miss-objective A/B evaluation is unaffected. End-to-end acceptance
of the processor option with a fresh training update remains pending.

GPU acceptance:

- All20 control corpora from the completed miss-objective collection were
  finalized again on CUDA, strictly as an audit without PPO training.
- For every case, `rollout.jsonl`, `sequence.jsonl` and
  `cuda-verification.json` are byte-exact against the sealed standalone output.
- Batch receipts additionally anchor request/entrypoint source hashes, so ready
  report bytes intentionally differ.
- The20-case function duration was75.62seconds under concurrent native pool
  load. This excludes the initial interpreter import, and is not a claim about
  total training or capture throughput.

Evidence: `workspace/artifacts/cuda-finalize-batch-audit-20261010/receipt/report.json`
and `content-verification.json`. A separate two-case comparison measures child
startup/import wall time for two independent processes versus one batch, on
identical input corpora with byte-exact output checks. Its result, when complete,
is `benchmark/report.json` under the same audit root. A single timing comparison
under heavy CPU load cannot establish general throughput.

For the next fresh corpus, invoke the processor with both CUDA flags; preserve
this mode on resume. Do not reinterpret already-consumed audit corpora as fresh
on-policy training data.

The two-case timing finished:26.28seconds for separate CUDA processes versus15.72seconds for one batch, ratio1.67; all three numerical/context/audit files remained byte-exact. This is one local measurement under concurrent evaluation load. The20-case acceptance covers the current Postmove temporal-attention actor/value; other architectures retain the existing finalizer contracts but were not benchmarked here.

The first fresh full-cycle collection has now completed80 own-policy fights
(train128..131). Its20 corpora were finalized successfully in one CUDA
interpreter under `first-life-update4-process-v2-20261010/control/cuda-finalize-batch`.
The batch receipt reports50.47seconds inside the batch function, excluding
interpreter/import startup. This is a different fresh workload, not a paired
speed comparison with the old75.62second workload. Merge, PPO update and exact
checkpoint audit follow; their completion is still pending at this entry.
The initial processing attempt had a death-stop replay metadata mismatch and
ended before PPO; recovery retained all native captures and failed artifacts.
See `combat_first_life_tail_20261010.md` for the repaired replay proof.

Full-cycle acceptance now passed:20/20 batch finalizations,5695-transition
merge, resumed CUDA PPO update4 with10 accepted actor steps, and exact CUDA
checkpoint/Adam restoration. Processing report:
`workspace/artifacts/first-life-update4-process-v2-20261010/report.json`.
The runtime quality comparison remains pending. The paired two-case benchmark
still measures only finalization speed; full-cycle acceptance does not establish
the same speedup for training or native gameplay.
