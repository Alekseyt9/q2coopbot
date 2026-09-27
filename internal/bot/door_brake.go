package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// Arrest neutral coast only outside the horizontal footprint of an observed
// vertically closing door. This does not predict arbitrary movers or escape
// overlap; the normal door guard must already have cancelled movement.
func brakeClosingDoorCoast(previous, s quake.Snapshot, cmd quake.UserCmd, g *quake.MapInfo) (quake.UserCmd, bool) {
	if g == nil || cmd.Msec != 100 || cmd.Forward != 0 || cmd.Side != 0 || cmd.Up != 0 || s.Ducked || previous.Map != s.Map || previous.Frame+1 != s.Frame || previous.Frame <= 0 || previous.Health <= 0 || quake.Distance(previous.Self, s.Self) > 64 || g.GroundFrictionStatus(s.Self) != "dry_flat" {
		return cmd, false
	}
	if s.Teammate != nil && quake.Distance(s.Self, *s.Teammate) < 96 {
		return cmd, false
	}
	for _, objects := range [][]quake.Object{s.Enemies, s.Obstacles} {
		for _, obj := range objects {
			if quake.Distance(s.Self, obj.Origin) < 96 {
				return cmd, false
			}
		}
	}
	prediction := predictGroundStep(s, cmd)
	if prediction == nil || math.Hypot(s.SelfVelocity[0], s.SelfVelocity[1]) < 50 {
		return cmd, false
	}
	model, reason := g.DoorMoveBlockStep(s.Movers, s.Self, s.SelfVelocity[0], s.SelfVelocity[1], prediction.NeutralStopDistance)
	if reason != "dynamic_door_blocked" || nearbyGroundBrushModelExcept(s, g, model) != 0 {
		return cmd, false
	}
	bounds, ok := g.Model(model)
	if !ok {
		return cmd, false
	}
	var current, old *quake.Mover
	for i := range s.Movers {
		if s.Movers[i].Model == model {
			current = &s.Movers[i]
		}
	}
	for i := range previous.Movers {
		if previous.Movers[i].Model == model {
			old = &previous.Movers[i]
		}
	}
	if current == nil || old == nil || current.ID != old.ID || current.Origin[0] != old.Origin[0] || current.Origin[1] != old.Origin[1] {
		return cmd, false
	}
	dz := old.Origin[2] - current.Origin[2]
	if !(dz > 0.125 && dz <= 32) {
		return cmd, false
	}
	// At least one horizontal separating axis with one unit of clearance:
	// vertical motion cannot reach the stationary standing hull.
	separated := false
	for axis := 0; axis < 2; axis++ {
		if s.Self[axis]+16 <= bounds.Min[axis]+current.Origin[axis]-1 || s.Self[axis]-16 >= bounds.Max[axis]+current.Origin[axis]+1 {
			separated = true
		}
	}
	if !separated {
		return cmd, false
	}
	speed := math.Hypot(prediction.Velocity[0], prediction.Velocity[1])
	if speed < 1 || speed > 300 {
		return cmd, false
	}
	brake := worldMove(cmd, s, -prediction.Velocity[0], -prediction.Velocity[1], speed, true)
	checked := diagnoseGroundStep(s, brake, g)
	if checked == nil || checked.CommandThenStopPath != "static_sampled_clear" || math.Hypot(checked.Displacement[0], checked.Displacement[1]) > 0.125 || math.Hypot(checked.Velocity[0], checked.Velocity[1]) > 1 {
		return cmd, false
	}
	return brake, true
}
