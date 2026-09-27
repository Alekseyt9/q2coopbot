package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// directCrouchRoute is a bounded, fully checked local connection to a visible
// teammate. It supplies low passages missing from the standing AAS graph.
func (p *Planner) directCrouchRoute(s quake.Snapshot, goal quake.Vec3) ([]quake.Waypoint, bool) {
	g := p.World.Geometry
	distance := quake.Horizontal(s.Self, goal)
	if p.World.Goal != "follow_teammate" || s.Teammate == nil || !s.OnGround ||
		!g.HasCollision() || distance < 80 || distance > 256 || math.Abs(goal[2]-s.Self[2]) > 1 ||
		g.PlayerMoveClear(s.Self, goal) {
		return nil, false
	}
	dx, dy := (goal[0]-s.Self[0])/distance, (goal[1]-s.Self[1])/distance
	for at := 0.0; at < distance; at += 16 {
		origin := quake.Vec3{s.Self[0] + dx*at, s.Self[1] + dy*at, s.Self[2]}
		step := math.Min(16, distance-at)
		if !g.CrouchStepClear(origin, dx, dy, step) {
			return nil, false
		}
		if _, reason := g.DoorMoveBlockStep(s.Movers, origin, dx, dy, step); reason != "" {
			return nil, false
		}
	}
	return []quake.Waypoint{{Position: goal, Kind: 2}}, true
}
