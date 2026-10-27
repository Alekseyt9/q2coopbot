package bot

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

func TestForcedDuckProbeRejectsOrdinarySessions(t *testing.T) {
	base := Config{Host: "127.0.0.1", CombatMode: "learned", CombatCapture: true, TestSynchronous: true,
		TestCombatOnly: true, CombatCheckedWallSlide: true, CombatExecutionContract: policy.StockWeaponsFireCombatExecution, TestForcedDuckProbeFrame: 335}
	for _, mutate := range []func(*Config){
		func(c *Config) { c.Host = "192.0.2.1" },
		func(c *Config) { c.TestSynchronous = false },
		func(c *Config) { c.TestCombatOnly = false },
		func(c *Config) { c.CombatCapture = false },
		func(c *Config) { c.TestForcedDuckProbeFrame = -2 },
		func(c *Config) { c.TestForcedDuckProbeFrame = 1000001 },
		func(c *Config) { c.TestCheckedGroundRequestProbeFrame = 335 },
		func(c *Config) { c.CombatExecutionContract = policy.LegacyCombatExecution },
	} {
		c := base
		mutate(&c)
		err := Run(context.Background(), c)
		if err == nil || !(strings.Contains(err.Error(), "forced duck probe") || strings.Contains(err.Error(), "checked combat execution contract")) {
			t.Fatal("invalid diagnostic reached runtime", err)
		}
	}
}

func TestForcedDuckProbePreservesReleaseAndOtherGuards(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires local BSP assets")
	}
	var f struct {
		Snapshot quake.Snapshot `json:"snapshot"`
		Proposed quake.UserCmd  `json:"proposed"`
		Applied  quake.UserCmd  `json:"applied"`
	}
	data, err := os.ReadFile("testdata/forced-duck-jail3-frame335.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	g, err := quake.LoadMap(root, f.Snapshot.Map)
	if err != nil {
		t.Fatal(err)
	}
	p := Planner{World: World{Geometry: &g}}
	guarded := f.Applied
	guarded.Buttons = 0 // A separately suppressed attack must stay suppressed.
	safety := []policy.Intervention{{Component: "movement", Reason: "static_hull_blocked"}}
	got, changes := p.diagnosticForcedDuckRequest(f.Snapshot, f.Proposed, guarded, safety)
	want := guarded
	want.Forward, want.Side = f.Proposed.Forward, f.Proposed.Side
	if got != want || got.Up != 0 || len(changes) != 1 {
		t.Fatal("model release or another command component changed", got, want, changes)
	}
	for _, reason := range []string{"no_ground_support", "unsupported_motion_guard", "dynamic_door_blocked", "static_laser_hazard", ""} {
		got, changes := p.diagnosticForcedDuckRequest(f.Snapshot, f.Proposed, guarded, []policy.Intervention{{Component: "movement", Reason: reason}})
		if got != guarded || len(changes) != 0 {
			t.Fatal("another guard bypassed", reason)
		}
	}
	standing := f.Snapshot
	standing.Ducked = false
	got, changes = p.diagnosticForcedDuckRequest(standing, f.Proposed, guarded, safety)
	if got != guarded || len(changes) != 0 {
		t.Fatal("unobserved pose admitted")
	}
}
