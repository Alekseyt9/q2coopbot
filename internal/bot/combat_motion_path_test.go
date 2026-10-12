package bot

import (
	"q2coopbot/internal/quake"
	"testing"
)

func recordedMotion(t *testing.T) (*Planner, quake.Snapshot, quake.UserCmd) {
	c := policyClient(t, "learned")
	s := c.planner.World.Snapshot
	s.Self = quake.Vec3{-47.875, -223.75, 24.125}
	s.PMStateKnown = true
	s.Gravity = 800
	s.DeltaAngles = [3]int16{0, 24576, 0}
	cmd := quake.UserCmd{Pitch: -703, Yaw: -23402, Forward: -141, Side: 316, Buttons: 1, Msec: 100}
	return c.planner, s, cmd
}

func TestCheckedWallMotionPreservesEntireRecordedCommand(t *testing.T) {
	p, s, cmd := recordedMotion(t)
	if !p.checkedCombatWallMotion(s, cmd) {
		t.Fatal("recorded static wall slide not certified")
	}
	got, changes := p.guardDirectCombatMotion(s, cmd, true)
	if got != cmd || len(changes) != 0 {
		t.Fatal("checked physical slide changed policy command", got, changes)
	}
	legacy, _ := p.guardDirectCombat(s, cmd)
	if legacy.Forward != 0 {
		t.Fatal("legacy comparison changed", legacy)
	}
}

func TestCheckedMotionChecksAllAlternativesAndRejectsUncertainty(t *testing.T) {
	p, s, cmd := recordedMotion(t)
	prediction := predictPmoveGround(s, cmd, p.World.Geometry)
	if !p.checkedCombatMotionPaths(s, cmd, prediction) {
		t.Fatal("clear path rejected")
	}
	bad := *prediction
	bad.Position = quake.Vec3{100000, 0, 24}
	bad.Segments = [][2]quake.Vec3{{s.Self, bad.Position}}
	prediction.SupportAmbiguous = true
	prediction.Alternative = &bad
	if p.checkedCombatMotionPaths(s, cmd, prediction) {
		t.Fatal("unsafe alternative ignored")
	}
	prediction.Alternative = nil
	if p.checkedCombatMotionPaths(s, cmd, prediction) {
		t.Fatal("missing alternative assumed safe")
	}
	s.CollisionBodies = []quake.Object{{ID: 8, Origin: s.Self, Solid: 2 | (3 << 5) | (7 << 10)}}
	if p.checkedCombatWallMotion(s, cmd) {
		t.Fatal("uncertain actor overlap approved")
	}
	s.CollisionBodies = nil
	s.Movers = []quake.Mover{{Model: 1}}
	if p.checkedCombatWallMotion(s, cmd) {
		t.Fatal("mover treated as stationary")
	}
}
