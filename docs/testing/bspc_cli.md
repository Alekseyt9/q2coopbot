# Historical BSPC reconstruction CLI regression checks

This test exercises the historical `bspc_reconstruction` target, not the default
reachability-capable `bspc` target. The reconstructed binary ships with placeholder pipelines that mirror
the control-flow of the historical tool. The script at
`workspace/tools/bspc/run_cli_modes.py` exercises every CLI mode against lightweight
fixtures so that automated pipelines can detect argument parsing or filesystem
regressions quickly.

## Running the smoke test

1. Build the historical executable (`cmake -S . -B workspace/build/aas -G Ninja` and `cmake --build workspace/build/aas --target bspc_reconstruction`).
2. Execute the harness:

   ```bash
   python workspace/tools/bspc/run_cli_modes.py --bspc /path/to/workspace/build/aas/workspace/tools/bspc/bspc_reconstruction
   ```

   The script emits a JSON summary that records the command line for each mode,
the detected timing messages, and the artifacts that were validated. The
workspace defaults to `workspace/build/test-output/bspc_cli/` and is recreated for every
run.

### Expected outputs and timing summaries

Each mode that performs compilation asserts that stdout includes a message
matching the legacy `"%5.0f seconds elapsed"` format. The generated
artifacts are compared byte-for-byte against golden baselines stored under
`tests/support/assets/bspc/golden/`:

| Mode     | Inputs                                     | Required outputs |
|----------|--------------------------------------------|------------------|
| map2bsp  | `tests/support/assets/bspc/simple_room.map` | `.bsp`, `.prt`, `.lin` |
| map2aas  | `tests/support/assets/bspc/simple_room.map` | `.aas`, `.bsp`, `.prt`, `.lin` |
| bsp2map  | `workspace/tools/dev_tools/assets/maps/2box4.bsp` | No files (logging only) |
| bsp2bsp  | `workspace/tools/dev_tools/assets/maps/2box4.bsp` | `.bsp`, `.prt`, `.lin` |
| bsp2aas  | `workspace/tools/dev_tools/assets/maps/2box4.bsp` | `.aas` |

Text artifacts normalise newline and path separators to keep the diff friendly
across platforms. When a mismatch is detected, a unified diff is written next
to the generated artifact (for example,
`workspace/build/test-output/bspc_cli/map2bsp/simple_room.prt.diff`).

## Comparing outputs to golden baselines

The smoke test now performs all baseline comparisons automatically. Binary
artifacts are stored as Base64 payloads (for example,
`tests/support/assets/bspc/golden/map2bsp/simple_room.bsp.base64`) so that the
repository does not need to ship literal `.bsp`/`.aas` blobs; the harness
decodes these payloads before performing byte-for-byte comparisons. The JSON
report emitted via `--json` can still be consumed by downstream tooling to
summarise which artifacts were validated, but manual `diff`/`cmp` runs are no
longer necessary unless you need additional diagnostics.
