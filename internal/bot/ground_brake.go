package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// brakeGroundCoast only arrests an otherwise neutral, 100ms ground command
// approaching a static wall or loss of flat support. It does not override
// intended movement or handle moving platforms, airborne motion or custom physics.
func brakeGroundCoast(s quake.Snapshot, cmd quake.UserCmd, g *quake.MapInfo) (quake.UserCmd, bool) {
	if cmd.Msec != 100 || cmd.Forward != 0 || cmd.Side != 0 || cmd.Up != 0 || s.Ducked || math.Hypot(s.SelfVelocity[0], s.SelfVelocity[1]) < 50 || g.GroundFrictionStatus(s.Self) != "dry_flat" {
		return cmd, false
	}
	if s.Teammate != nil && quake.Distance(s.Self, *s.Teammate) < 96 {
		return cmd, false
	}
	for _, e := range s.Obstacles {
		if quake.Distance(s.Self, e.Origin) < 96 {
			return cmd, false
		}
	}
	for _, e := range s.Enemies {
		if quake.Distance(s.Self, e.Origin) < 96 {
			return cmd, false
		}
	}
	if nearbyGroundBrushModel(s, g) != 0 {
		return cmd, false
	}
	p := diagnoseGroundStep(s, cmd, g)
	if p == nil || (p.NeutralPath != "static_hull_blocked" && p.NeutralPath != "uneven_or_missing_support") {
		return cmd, false
	}
	speed := math.Hypot(p.Velocity[0], p.Velocity[1])
	if speed > 300 || speed < 1 {
		return cmd, false
	}
	// At dt=.1 and accel=10, an opposite wish speed equal to the velocity
	// after friction cancels it. Recheck after quantizing command components.
	brake := worldMove(cmd, s, -p.Velocity[0], -p.Velocity[1], speed, true)
	checked := diagnoseGroundStep(s, brake, g)
	if checked == nil || checked.CommandThenStopPath != "static_sampled_clear" || math.Hypot(checked.Displacement[0], checked.Displacement[1]) > 0.125 || math.Hypot(checked.Velocity[0], checked.Velocity[1]) > 1 {
		return cmd, false
	}
	return brake, true
}
