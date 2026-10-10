# Shared harness binaries — 2026-10-10

The native pool now builds the Go companion and exporter once per harness
source fingerprint. Each battle receives hardlinks to the sealed bundle, with
copy fallback on another volume. The receipt pins both executable hashes and
source records; existing destinations and mismatched bundles are rejected.
Native/runtime receipts still validate actual executable bytes for each member.
Shared PAK/AAS conventions remain in force.

Evidence: `workspace/artifacts/harness-bundle-contract-20261010/report.json`
checks reuse, hash identity, wrong fingerprint, corruption and destination
preservation. `workspace/artifacts/harness-bundle-native-20261010` sealed
16/16 native battles with unchanged sources, verified all32 member executables
as hardlinks and retained eight wins, seven deaths, mean damage received67.25.

Wall time was53.70 seconds previously and57.25 seconds with the bundle during
concurrent CUDA work. Mean time outside the measured native harness dropped
from15.55 to9.12 seconds, including build/wrapper/IO overhead. These samples
do not establish an overall throughput improvement. The storage advantage
is binary sharing for future captures; old copies have not been deduplicated.
