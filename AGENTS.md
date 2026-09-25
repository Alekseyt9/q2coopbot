# Repository guidance

This repository's active product is the Go Quake II UDP companion. Its CLI lives in `cmd/q2coopbot`, gameplay and session logic in `internal/bot`, and Quake protocol/BSP/AAS code in `internal/quake`. Keep gameplay decisions and protocol handling in Go. The Python harness under `examples/python-harness/` is a historical reference, not a second implementation target.

`workspace/tools/bspc/` is tracked source for the optional standalone AAS preparation utility. `workspace/build/`, `workspace/runtime/`, `workspace/artifacts/`, and `workspace/tools/dev_tools/` contain local generated or historical material and are ignored by Git. Keep the AAS utility buildable independently of the removed Gladiator Botlib and game module. Do not add game assets or generated binaries to Git.

Preserve `docs/` as the development record. Label obsolete C-bot instructions as historical when touching them. Validate Go changes with `go test ./...` and the relevant live UDP scenario; validate AAS utility changes with its CMake build and focused tests. Do not commit or push unless asked.
