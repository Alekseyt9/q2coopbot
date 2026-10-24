package bot

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"q2coopbot/internal/quake"
)

func TestStoppedCrouchProbeRejectsOrdinarySessions(t *testing.T) {
	valid := Config{Host: "127.0.0.1", CombatMode: "learned", CombatCapture: true, TestSynchronous: true,
		TestCombatOnly: true, TestStoppedCrouchProbeFrame: 104}
	for _, cfg := range []Config{
		{TestStoppedCrouchProbeFrame: -1},
		{TestStoppedCrouchProbeFrame: 1000001},
		{TestStoppedCrouchProbeAllFrames: true},
		func() Config { c := valid; c.TestStoppedCrouchProbeAllFrames = true; return c }(),
		func() Config {
			c := valid
			c.TestStoppedCrouchProbeFrame = 0
			c.TestStoppedCrouchProbeAllFrames = true
			c.Host = "192.0.2.1"
			return c
		}(),
		func() Config { c := valid; c.TestSynchronous = false; return c }(),
		func() Config { c := valid; c.TestCombatOnly = false; return c }(),
		func() Config { c := valid; c.CombatCapture = false; return c }(),
		func() Config { c := valid; c.CombatMode = "rules"; return c }(),
		func() Config { c := valid; c.Host = "192.0.2.1"; return c }(),
	} {
		err := Run(context.Background(), cfg)
		if err == nil || !strings.Contains(err.Error(), "stopped crouch probe") {
			t.Fatal("diagnostic reached normal session setup", err)
		}
	}
}

func TestStoppedCrouchProbeRestoresOnlyChosenPose(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires local BSP assets")
	}
	var frozen struct {
		Snapshot quake.Snapshot `json:"snapshot"`
		Proposed quake.UserCmd  `json:"proposed"`
		Applied  quake.UserCmd  `json:"applied"`
	}
	b, err := os.ReadFile("testdata/stopped-crouch-city1-frame104.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &frozen); err != nil {
		t.Fatal(err)
	}
	g, err := quake.LoadMap(root, frozen.Snapshot.Map)
	if err != nil {
		t.Fatal(err)
	}
	p := Planner{World: World{Geometry: &g}}
	got, changes := p.diagnosticStoppedCrouch(frozen.Snapshot, frozen.Proposed, frozen.Applied)
	expected := frozen.Applied
	expected.Up = frozen.Proposed.Up
	if got != expected || len(changes) != 1 || changes[0].Component != "vertical" || changes[0].Reason != "diagnostic_checked_stopped_crouch" {
		t.Fatal("checked recorded crouch not restored exactly", got, changes)
	}
	for _, variant := range []struct {
		s quake.Snapshot
		r quake.UserCmd
		a quake.UserCmd
	}{
		{func() quake.Snapshot { s := frozen.Snapshot; s.OnGround = false; return s }(), frozen.Proposed, frozen.Applied},
		{func() quake.Snapshot { s := frozen.Snapshot; s.Health = 0; return s }(), frozen.Proposed, frozen.Applied},
		{frozen.Snapshot, func() quake.UserCmd { c := frozen.Proposed; c.Up = 400; return c }(), frozen.Applied},
		{frozen.Snapshot, frozen.Proposed, func() quake.UserCmd { c := frozen.Applied; c.Forward = 1; return c }()},
		{frozen.Snapshot, frozen.Proposed, func() quake.UserCmd { c := frozen.Applied; c.Side = 1; return c }()},
		{frozen.Snapshot, frozen.Proposed, expected},
	} {
		got, changes = p.diagnosticStoppedCrouch(variant.s, variant.r, variant.a)
		if got != variant.a || len(changes) != 0 {
			t.Fatal("probe modified an ineligible command", got, changes)
		}
	}
}
