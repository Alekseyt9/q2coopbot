package bot

import (
	"path/filepath"
	"q2coopbot/internal/harness"
	"testing"
)

func TestSessionStepSignalRejectsStaleIdentity(t *testing.T) {
	for _, kind := range []string{"valid", "map", "generation", "phase", "role", "stale", "future"} {
		t.Run(kind, func(t *testing.T) {
			c := Client{session: &harness.SessionRunner{Status: harness.SessionStatus{Phase: harness.Status{State: "running"}}}, sessionDefinition: &harness.Session{Phases: []harness.Phase{{Scenario: harness.Scenario{Steps: []harness.Step{{ID: "release-player", Action: "wait_signal"}}}}}}, sessionMap: "base2", spawncount: 7, sessionStartFrame: 64, latestFrame: 100, scenarioResultPath: filepath.Join(t.TempDir(), "completion.json")}
			signal := sessionReady{Map: "base2", Generation: 7, Phase: 0, Frame: 100, Role: "observer"}
			switch kind {
			case "map":
				signal.Map = "base1"
			case "generation":
				signal.Generation = 8
			case "phase":
				signal.Phase = 1
			case "role":
				signal.Role = "actor"
			case "stale":
				signal.Frame = 63
			case "future":
				signal.Frame = 101
			}
			path := filepath.Join(c.scenarioResultPath+".barrier", "0-7-release-player-signal.json")
			if err := writeSessionSignal(path, signal); err != nil {
				t.Fatal(err)
			}
			id, err := c.sessionStepSignal()
			if kind == "valid" {
				if err != nil || id != "release-player" {
					t.Fatal(id, err)
				}
			} else if kind == "future" {
				if err != nil || id != "" {
					t.Fatal("future signal did not wait", id, err)
				}
				c.latestFrame++
				if id, err = c.sessionStepSignal(); err != nil || id == "" {
					t.Fatal("future signal never released", id, err)
				}
			} else if err == nil || id != "" {
				t.Fatal("stale identity accepted", id, err)
			}
		})
	}
}
