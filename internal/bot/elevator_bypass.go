package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

func (p *Planner) elevatorBypassClear(s quake.Snapshot, mover quake.Mover, from, to quake.Vec3) bool {
	g := p.World.Geometry
	if g == nil || !g.PlayerMoveClear(from, to) || !g.MoverHullClear(mover, from, to) {
		return false
	}
	length := quake.Horizontal(from, to)
	if length > 100 || math.Abs(from[2]-to[2]) > 0.25 {
		return false
	}
	if _, reason := g.DoorMoveBlockStep(s.Movers, from, to[0]-from[0], to[1]-from[1], length); reason != "" {
		return false
	}
	for _, other := range s.Movers {
		if other.Model != mover.Model && !g.MoverHullClear(other, from, to) {
			return false
		}
	}
	for step := 0.; step <= math.Ceil(length); step++ {
		fraction := 0.
		if length > 0 {
			fraction = math.Min(step/length, 1)
		}
		point := from
		for axis := 0; axis < 2; axis++ {
			point[axis] += (to[axis] - from[axis]) * fraction
		}
		if s.Teammate != nil && math.Abs(point[2]-s.Teammate[2]) < 56 && math.Abs(point[0]-s.Teammate[0]) < 32.05 && math.Abs(point[1]-s.Teammate[1]) < 32.05 {
			return false
		}
		for _, object := range s.Obstacles {
			if math.Abs(point[2]-object.Origin[2]) < 64 && quake.Horizontal(point, object.Origin) < 48 {
				return false
			}
		}
		for _, offset := range []quake.Vec3{{}, {-16, -16, 0}, {-16, 16, 0}, {16, -16, 0}, {16, 16, 0}} {
			probe := point
			probe[0] += offset[0]
			probe[1] += offset[1]
			if !elevatorSupportedPoint(g, mover, probe) {
				return false
			}
		}
	}
	return true
}

func elevatorSupportedPoint(g *quake.MapInfo, mover quake.Mover, point quake.Vec3) bool {
	grounded := func(p quake.Vec3) bool {
		if drop, ok := g.GroundDrop(p, 0.5); ok && drop <= 0.5 {
			return true
		}
		drop, ok := g.MoverFooting(mover, p, 0.5)
		return ok && drop <= 0.5
	}
	if grounded(point) {
		return true
	}
	// Exact shared brush edges can reject a point trace. Require support
	// on all four sides within one network coordinate quantum, not one side.
	for _, offset := range []quake.Vec3{{-.125, -.125, 0}, {.125, -.125, 0}, {-.125, .125, 0}, {.125, .125, 0}} {
		probe := point
		probe[0] += offset[0]
		probe[1] += offset[1]
		if !grounded(probe) {
			return false
		}
	}
	return true
}
func (p *Planner) elevatorBypass(cmd quake.UserCmd, s quake.Snapshot, target quake.Vec3, model quake.BSPModel, mover quake.Mover) (quake.UserCmd, bool) {
	if p.elevator == nil || !s.OnGround || s.Teammate == nil || math.Abs(mover.Origin[2]-model.Origin[2]) > 0.125 || math.Abs(s.SelfVelocity[2]) > 1 {
		return cmd, false
	}
	ride := p.elevator
	if len(ride.bypass) == 0 {
		distance := quake.Horizontal(s.Self, target)
		if distance < 1 {
			return cmd, false
		}
		dx, dy := (target[0]-s.Self[0])/distance, (target[1]-s.Self[1])/distance
		for _, sign := range []float64{1, -1} {
			side := s.Self
			side[0] -= dy * 40 * sign
			side[1] += dx * 40 * sign
			end := side
			end[0] += dx * 60
			end[1] += dy * 60
			if p.elevatorBypassClear(s, mover, s.Self, side) && p.elevatorBypassClear(s, mover, side, end) {
				ride.bypass = []quake.Vec3{side, end}
				break
			}
		}
	}
	for len(ride.bypass) > 0 && quake.Horizontal(s.Self, ride.bypass[0]) <= 4 {
		ride.bypass = ride.bypass[1:]
	}
	if len(ride.bypass) == 0 {
		return cmd, false
	}
	next := ride.bypass[0]
	if !p.elevatorBypassClear(s, mover, s.Self, next) {
		ride.bypass = nil
		return cmd, false
	}
	p.World.Elevator = "exit_teammate_bypass"
	cmd.Up = 0
	return worldMove(cmd, s, next[0]-s.Self[0], next[1]-s.Self[1], 60, false), true
}
