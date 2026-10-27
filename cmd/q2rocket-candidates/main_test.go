package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRocketCandidateRequestRejectsFutureAndNativeState(t *testing.T) {
	for _, extra := range []string{`"next_observation":{}`, `"reward":1`, `"monster_health":100`} {
		t.Run(extra, func(t *testing.T) {
			dir := t.TempDir()
			request := filepath.Join(dir, "request.json")
			out := filepath.Join(dir, "out.json")
			body := `{"version":"combat_observed_rocket_candidate_request_v1","runtime_root":"unused","entries":[{"id":"x","observation":{` + extra + `}}]}`
			if err := os.WriteFile(request, []byte(body), 0600); err != nil {
				t.Fatal(err)
			}
			if err := run(request, out); err == nil || !strings.Contains(err.Error(), "unknown field") {
				t.Fatal("future/native field admitted", err)
			}
			if _, err := os.Stat(out); !os.IsNotExist(err) {
				t.Fatal("invalid request published output")
			}
		})
	}
}
