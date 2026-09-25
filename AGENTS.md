# Repository guidance

This repository's active product is the Go Quake II UDP companion at the repository root. Keep gameplay decisions and protocol handling in Go. The Python harness under `examples/python-harness/` is a historical reference, not a second implementation target.

`tools/bspc/` is an optional standalone AAS preparation utility. Keep it buildable independently of the removed Gladiator Botlib and game module. Do not add game assets or generated binaries to Git.

Preserve `docs/` as the development record. Label obsolete C-bot instructions as historical when touching them. Validate Go changes with `go test ./...` and the relevant live UDP scenario; validate AAS utility changes with its CMake build and focused tests. Do not commit or push unless asked.
