package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// A coop spawn can be disconnected from AAS even on a flat open floor.
// Permit only a bounded ground walk whose complete hull, support and movers
// are checked; this never grants a jump or an unchecked search shortcut.
func (p *Planner) supportedReturnRoute(s quake.Snapshot, goal quake.Vec3) ([]quake.Waypoint, bool) {
	g := p.World.Geometry
	d := quake.Horizontal(s.Self, goal)
	if !s.OnGround || g == nil || !g.HasCollision() || p.Nav == nil || d < 1 || d > 512 || math.Abs(goal[2]-s.Self[2]) > 1 {
		return nil, false
	}
	dx, dy := (goal[0]-s.Self[0])/d, (goal[1]-s.Self[1])/d
	var route []quake.Waypoint
	from := s.Self
	for at := 0.0; at < d; at += 16 {
		step := math.Min(16, d-at)
		to := quake.Vec3{from[0] + dx*step, from[1] + dy*step, from[2]}
		if !g.PlayerMoveClear(from, to) || g.GroundMoveHazardStep(p.Nav, from, dx, dy, step) != "" {
			return nil, false
		}
		for sample := 0.0; sample <= step; sample += 2 {
			at := quake.Vec3{from[0] + dx*sample, from[1] + dy*sample, from[2]}
			if _, ok := g.GroundDrop(at, 18); !ok && !p.Nav.GroundedNear(at) {
				return nil, false
			}
		}
		if _, reason := g.DoorMoveBlockStep(s.Movers, from, dx, dy, step); reason != "" {
			return nil, false
		}
		for _, mover := range s.Movers {
			if !g.MoverHullClear(mover, from, to) {
				return nil, false
			}
		}
		route = append(route, quake.Waypoint{Position: to, Kind: 2})
		from = to
	}
	return route, true
}
