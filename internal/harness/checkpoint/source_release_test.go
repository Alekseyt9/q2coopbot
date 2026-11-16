package checkpoint

import (
	"encoding/json"
	"path/filepath"
	"testing"
)

func TestCombatBoundaryCannotMasqueradeAsLegacyPlanner(t *testing.T) {
	r := CaptureRequest{Version: 1, ID: "combat", Map: "base1", Frame: 123, Generation: 7, Boundary: CombatPreDecisionBoundary}
	c := Capture{CaptureRequest: r, Participant: "Bot", SelfEntity: 1, Health: 99, Combat: json.RawMessage(`{"version":"hjepa_predecision_checkpoint_v1","decision_root":{"map":"base1","frame":123,"spawncount":7,"actor":1}}`)}
	if err := validateCapture(c, r); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*Capture){
		func(c *Capture) { c.Planner = json.RawMessage(`{"version":1}`) },
		func(c *Capture) { c.SelfEntity = 2 },
		func(c *Capture) { c.Combat = nil },
		func(c *Capture) { c.Boundary = "" },
	} {
		bad := c
		mutate(&bad)
		if validateCapture(bad, r) == nil {
			t.Fatal("mixed or mismatched boundary admitted")
		}
	}
	root := t.TempDir()
	proof, err := saveCaptures(root, &Barrier{Boundary: r.Boundary, ID: r.ID, Map: r.Map, Frame: r.Frame, Generation: r.Generation}, []Capture{c})
	if err != nil {
		t.Fatal(err)
	}
	if err := verifySavedCaptures(root, r.Map, proof); err != nil {
		t.Fatal(err)
	}
	proof.Boundary = ""
	if err := verifySavedCaptures(root, r.Map, proof); err == nil {
		t.Fatal("boundary provenance lost")
	}
	path := filepath.Join(t.TempDir(), "released")
	release := SourceRelease{CaptureRequest: r, State: "native_released"}
	if err := WriteCapture(path, release); err != nil {
		t.Fatal(err)
	}
	if ready, err := SourceReleaseReady(path, r); err != nil || !ready {
		t.Fatal(ready, err)
	}
	r.Generation++
	if _, err := SourceReleaseReady(path, r); err == nil {
		t.Fatal("different generation admitted")
	}
}
