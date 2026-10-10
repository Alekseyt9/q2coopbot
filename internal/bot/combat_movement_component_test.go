package bot

import (
	"testing"

	"q2coopbot/internal/quake"
)

// Native loss seed 730024, frame 115: diagonal retreat pushes into the
// west wall although the requested strafe remains available.
func TestCombatMovementComponentAtRecordedWall(t *testing.T) {
	c := policyClient(t, "learned")
	s := c.planner.World.Snapshot
	s.Self = quake.Vec3{-47.875, -223.75, 24.125}
	s.SelfVelocity = quake.Vec3{}
	s.DeltaAngles = [3]int16{0, 24576, 0}
	cmd := quake.UserCmd{Pitch: -703, Yaw: -23402, Forward: -141, Side: 316, Buttons: 1, Msec: 100}
	prediction := predictGroundStep(s, cmd)
	g := c.planner.World.Geometry
	if g.GroundMoveHazardStep(nil, s.Self, prediction.Displacement[0], prediction.Displacement[1], 30) != "static_hull_blocked" {
		t.Fatal("recorded obstruction was not reproduced")
	}
	got, changes := c.planner.guardDirectCombat(s, cmd)
	if got.Forward != 0 || got.Side != cmd.Side || got.Yaw != cmd.Yaw || got.Pitch != cmd.Pitch || got.Buttons != cmd.Buttons || len(changes) != 1 || changes[0].Reason != "static_hull_component_clipped" {
		t.Fatal("safe strafe or policy aim/fire changed", got, changes)
	}
	for _, hazard := range []string{"no_ground_support", "static_laser_hazard", "bsp_unavailable"} {
		if _, ok := c.planner.checkedCombatMovementComponent(s, cmd, hazard); ok {
			t.Fatal("hazard bypass", hazard)
		}
	}
	for _, changed := range []quake.Snapshot{
		func() quake.Snapshot { x := s; x.OnGround = false; return x }(),
		func() quake.Snapshot { x := s; x.SelfVelocity = quake.Vec3{-300, 0, 0}; return x }(),
	} {
		if _, ok := c.planner.checkedCombatMovementComponent(changed, cmd, "static_hull_blocked"); ok {
			t.Fatal("unsupported motion or residual wallward inertia admitted", changed)
		}
	}
	cmd.Side = 0
	got, _ = c.planner.guardDirectCombat(s, cmd)
	if got.Forward != 0 || got.Side != 0 {
		t.Fatal("head-on command acquired an invented strafe", got)
	}
}
