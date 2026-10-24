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

func TestCheckedGroundProbeRejectsOrdinarySessions(t *testing.T) {
	base := Config{Host: "127.0.0.1", CombatMode: "learned", CombatCapture: true, TestSynchronous: true,
		TestCombatOnly: true, TestCheckedGroundRequestProbeFrame: 104}
	for _, mutate := range []func(*Config){
		func(c *Config) { c.TestCheckedGroundRequestProbeFrame = -1 },
		func(c *Config) { c.TestCheckedGroundRequestProbeFrame = 1000001 },
		func(c *Config) { c.TestSynchronous = false },
		func(c *Config) { c.TestCombatOnly = false },
		func(c *Config) { c.CombatCapture = false },
		func(c *Config) { c.CombatMode = "rules" },
		func(c *Config) { c.Host = "192.0.2.1" },
		func(c *Config) { c.TestStoppedCrouchProbeFrame = 104 },
		func(c *Config) { c.TestStoppedCrouchProbeAllFrames = true },
		func(c *Config) { c.TestCheckedGroundRequestProbeAllFrames = true },
		func(c *Config) {
			c.TestCheckedGroundRequestProbeFrame = 0
			c.TestCheckedGroundRequestProbeAllFrames = true
			c.TestSynchronous = false
		},
		func(c *Config) {
			c.TestCheckedGroundRequestProbeFrame = 0
			c.TestCheckedGroundRequestProbeAllFrames = true
			c.CombatMode = "rules"
		},
	} {
		c := base
		mutate(&c)
		if err := Run(context.Background(), c); err == nil || !strings.Contains(err.Error(), "checked ground request probe") {
			t.Fatal("invalid probe reached normal runtime", err)
		}
	}
}

func TestCheckedGroundProbeRestoresOnlyVerifiedMovement(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires local BSP")
	}
	for _, name := range []string{"stopped-crouch-city1-frame104.json", "checked-ground-contact-city1-frame138.json"} {
		var fixture struct {
			Snapshot quake.Snapshot `json:"snapshot"`
			Proposed quake.UserCmd  `json:"proposed"`
		}
		data, err := os.ReadFile("testdata/" + name)
		if err != nil {
			t.Fatal(err)
		}
		if err = json.Unmarshal(data, &fixture); err != nil {
			t.Fatal(err)
		}
		g, err := quake.LoadMap(root, fixture.Snapshot.Map)
		if err != nil {
			t.Fatal(err)
		}
		p := Planner{World: World{Geometry: &g}}
		guarded, safety := p.guardDirectCombatMotion(fixture.Snapshot, fixture.Proposed, true)
		if guarded.Forward != 0 || guarded.Side != 0 || guarded.Up != 0 {
			t.Fatal("fixture did not reach unsupported stop")
		}
		// An attack safety veto and the protocol light must survive restoring movement.
		guarded.Buttons &^= 1
		guarded.Light = 69
		got, changes := p.diagnosticCheckedGroundRequest(fixture.Snapshot, fixture.Proposed, guarded, safety)
		expected := guarded
		expected.Forward, expected.Side, expected.Up = fixture.Proposed.Forward, fixture.Proposed.Side, fixture.Proposed.Up
		if got != expected || len(changes) != 1 || changes[0].Reason != "diagnostic_checked_ground_request" {
			t.Fatal("requested movement not restored exactly", got, changes)
		}
		for _, reason := range []string{"", "static_hull_blocked", "no_ground_support", "static_laser_hazard", "dynamic_door_blocked"} {
			bad := []policy.Intervention{{Component: "movement", Reason: reason}}
			got, changes = p.diagnosticCheckedGroundRequest(fixture.Snapshot, fixture.Proposed, guarded, bad)
			if got != guarded || len(changes) != 0 {
				t.Fatal("probe bypassed a different safety reason", reason)
			}
		}
		for _, mutate := range []func(*quake.Snapshot, *quake.UserCmd){
			func(s *quake.Snapshot, c *quake.UserCmd) { s.OnGround = false },
			func(s *quake.Snapshot, c *quake.UserCmd) { s.Health = 0 },
			func(s *quake.Snapshot, c *quake.UserCmd) { s.PMStateKnown = false },
			func(s *quake.Snapshot, c *quake.UserCmd) { s.SelfVelocity[2] = 181 },
			func(s *quake.Snapshot, c *quake.UserCmd) { c.Up = 400 },
			func(s *quake.Snapshot, c *quake.UserCmd) { c.Forward = 0; c.Side = 0 },
		} {
			s, cmd := fixture.Snapshot, fixture.Proposed
			mutate(&s, &cmd)
			got, changes = p.diagnosticCheckedGroundRequest(s, cmd, guarded, safety)
			if got != guarded || len(changes) != 0 {
				t.Fatal("unsupported state admitted", got, changes)
			}
		}
	}
}
