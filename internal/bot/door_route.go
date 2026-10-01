package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// Open sliding doors can occupy the old AAS walking reaches. Use a later
// walking point only if its entire short, level approach is supported and
// clears both static geometry and the currently observed moving brushes.
func (p *Planner) doorRouteBypass(s quake.Snapshot) (quake.Vec3, bool) {
	g := p.World.Geometry
	if !s.OnGround || !g.HasCollision() || (p.World.Goal != "follow_teammate" && p.World.Goal != "regroup_after_respawn" && p.World.Goal != "reach_level_exit") {
		return quake.Vec3{}, false
	}
	for i, wp := range p.World.Route {
		if wp.Kind != 2 || wp.Jump {
			break
		}
		d := quake.Horizontal(s.Self, wp.Position)
		if i == 0 || d < 24 || d > 192 || math.Abs(wp.Position[2]-s.Self[2]) > 2 {
			continue
		}
		ends := []quake.Vec3{wp.Position, {wp.Position[0], s.Self[1], s.Self[2]}, {s.Self[0], wp.Position[1], s.Self[2]}}
		for _, end := range ends {
			end[2] = s.Self[2]
			d = quake.Horizontal(s.Self, end)
			if d < 16 {
				continue
			}
			if d > 64 {
				end[0] = s.Self[0] + (end[0]-s.Self[0])*64/d
				end[1] = s.Self[1] + (end[1]-s.Self[1])*64/d
				d = 64
			}
			if !g.PlayerMoveClear(s.Self, end) {
				continue
			}
			if _, reason := g.DoorMoveBlockStep(s.Movers, end, wp.Position[0]-end[0], wp.Position[1]-end[1], quake.Horizontal(end, wp.Position)); reason != "" {
				continue
			}
			clear := true
			for _, m := range s.Movers {
				// A sideways alignment must lead past the obstruction,
				// otherwise the next frame can choose the opposite side again.
				if !g.MoverHullClear(m, s.Self, end) || !g.MoverHullClear(m, end, wp.Position) {
					clear = false
					break
				}
			}
			if !clear {
				continue
			}
			for _, o := range s.Obstacles {
				dx, dy := end[0]-s.Self[0], end[1]-s.Self[1]
				t := math.Max(0, math.Min(1, ((o.Origin[0]-s.Self[0])*dx+(o.Origin[1]-s.Self[1])*dy)/(d*d)))
				nearest := quake.Vec3{s.Self[0] + t*dx, s.Self[1] + t*dy, s.Self[2]}
				if math.Abs(o.Origin[2]-s.Self[2]) < 48 && quake.Horizontal(o.Origin, nearest) < 36 {
					clear = false
					break
				}
			}
			if !clear {
				continue
			}
			at := s.Self
			steps := int(math.Ceil(d / 8))
			for j := 1; j <= steps; j++ {
				t := float64(j) / float64(steps)
				next := quake.Vec3{s.Self[0] + (end[0]-s.Self[0])*t, s.Self[1] + (end[1]-s.Self[1])*t, s.Self[2]}
				if _, ok := g.GroundDrop(next, 18); !ok {
					clear = false
					break
				}
				if _, reason := g.DoorMoveBlockStep(s.Movers, at, next[0]-at[0], next[1]-at[1], quake.Horizontal(at, next)); reason != "" {
					clear = false
					break
				}
				at = next
			}
			if clear {
				return end, true
			}
		}
	}
	return quake.Vec3{}, false
}
