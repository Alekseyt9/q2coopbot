package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// localFlatRoute reconnects an unclassified position to a nearby AAS area.
// The new short segment needs standing headroom, static floor and clear doors.
func (p *Planner) localFlatRoute(s quake.Snapshot, goal quake.Vec3) ([]quake.Waypoint, bool) {
	g := p.World.Geometry
	d := quake.Horizontal(s.Self, goal)
	if p.World.Goal != "follow_teammate" || s.Teammate == nil || !s.OnGround || p.Nav == nil || !g.HasCollision() || d < 80 || d > 256 || math.Abs(goal[2]-s.Self[2]) > 1 {
		return nil, false
	}
	ux, uy := (goal[0]-s.Self[0])/d, (goal[1]-s.Self[1])/d
	clear := func(from, to quake.Vec3) bool {
		length := quake.Horizontal(from, to)
		if length < 0.001 {
			return false
		}
		dx, dy := (to[0]-from[0])/length, (to[1]-from[1])/length
		for at := 0.0; at < length; at += 16 {
			a := quake.Vec3{from[0] + dx*at, from[1] + dy*at, from[2]}
			step := math.Min(16, length-at)
			b := quake.Vec3{a[0] + dx*step, a[1] + dy*step, a[2]}
			if !g.PlayerMoveClear(a, b) || !g.CrouchStepClear(a, dx, dy, step) {
				return false
			}
			if _, reason := g.DoorMoveBlockStep(s.Movers, a, dx, dy, step); reason != "" {
				return false
			}
		}
		return true
	}
	for _, offset := range []float64{16, -16, 32, -32} {
		a := s.Self
		a[0] -= uy * offset
		a[1] += ux * offset
		if !clear(s.Self, a) || !p.Nav.GroundedNear(a) {
			continue
		}
		route, ok := p.Nav.Route(a, goal)
		if !ok || len(route) == 0 {
			continue
		}
		flat := true
		for _, wp := range route {
			if wp.Kind != 2 || math.Abs(wp.Position[2]-s.Self[2]) > 18 {
				flat = false
				break
			}
		}
		if flat {
			return append([]quake.Waypoint{{Position: a, Kind: 2}}, route...), true
		}
	}
	return nil, false
}
