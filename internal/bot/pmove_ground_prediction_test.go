package bot

import (
	"math"
	"q2coopbot/internal/quake"
	"testing"
)

func TestPmoveGroundUsesVerticalSpeedBeforeFriction(t *testing.T) {
	c := policyClient(t, "learned")
	s := c.planner.World.Snapshot
	s.PMStateKnown = true
	s.Gravity = 800
	s.SelfVelocity = quake.Vec3{50, 0, 80}
	p := predictPmoveGround(s, quake.UserCmd{Msec: 100}, c.planner.World.Geometry)
	expected := 50 * (1 - 60/math.Sqrt(50*50+80*80))
	if p == nil || math.Abs(p.Velocity[0]-expected) > 1e-8 || p.Velocity[2] != 0 {
		t.Fatal("3D friction lost", p, expected)
	}
	for _, mutate := range []func(*quake.Snapshot){
		func(s *quake.Snapshot) { s.PMTime = 7 }, func(s *quake.Snapshot) { s.PMType = 2 },
		func(s *quake.Snapshot) { s.PMStateKnown = false }, func(s *quake.Snapshot) { s.OnGround = false },
	} {
		bad := s
		mutate(&bad)
		if predictPmoveGround(bad, quake.UserCmd{Msec: 100}, c.planner.World.Geometry) != nil {
			t.Fatal("unsupported state predicted", bad)
		}
	}
}

func TestPmoveGroundRetainsRecordedWallTrajectory(t *testing.T) {
	c := policyClient(t, "learned")
	s := c.planner.World.Snapshot
	s.Self = quake.Vec3{-47.875, -223.75, 24.125}
	s.PMStateKnown = true
	s.Gravity = 800
	s.DeltaAngles = [3]int16{0, 24576, 0}
	cmd := quake.UserCmd{Pitch: -703, Yaw: -23402, Forward: -141, Side: 316, Msec: 100}
	p := predictPmoveGround(s, cmd, c.planner.World.Geometry)
	if p == nil || len(p.Contacts) == 0 || math.Hypot(p.Position[0]-s.Self[0], p.Position[1]-s.Self[1]) < .125 {
		t.Fatal("native wall slide missing", p)
	}
}
