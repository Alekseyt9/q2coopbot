package bot

import (
	"math"

	"q2coopbot/internal/quake"
)

// regroupCornerStep makes one short, grounded sidestep around a blocked AAS
// walking corner. The next snapshot replans from the observed new position.
func (p *Planner) regroupCornerStep(s quake.Snapshot, target quake.Vec3) (float64, float64, bool) {
	if p.World.Goal != "regroup_after_respawn" || !s.OnGround || p.Nav == nil || p.World.Geometry == nil || !p.World.Geometry.HasCollision() ||
		len(p.World.Route) == 0 || p.World.Route[0].Kind != 2 || quake.Horizontal(s.Self, target) > 64 {
		return 0, 0, false
	}
	if p.blockedRiseApproach() && p.World.Route[0].Position[2]-s.Self[2] < 8 {
		// A distant rise is a jump/ramp problem, not this local corner.
		return 0, 0, false
	}
	dx, dy := target[0]-s.Self[0], target[1]-s.Self[1]
	length := math.Hypot(dx, dy)
	if length < 4 {
		return 0, 0, false
	}
	ux, uy := dx/length, dy/length
	g := p.World.Geometry
	for _, turn := range []float64{math.Pi / 3, -math.Pi / 3, math.Pi / 2, -math.Pi / 2} {
		x, y := 16*(ux*math.Cos(turn)-uy*math.Sin(turn)), 16*(ux*math.Sin(turn)+uy*math.Cos(turn))
		end := s.Self
		end[0] += x
		end[1] += y
		if g.GroundMoveHazardStep(p.Nav, s.Self, x, y, 16) != "" || g.DoorMoveHazard(s.Movers, s.Self, x, y) != "" || !g.PlayerMoveClear(end, end) {
			continue
		}
		blocked := false
		for _, mover := range s.Movers {
			if !g.MoverHullClear(mover, s.Self, end) {
				blocked = true
				break
			}
		}
		if blocked {
			continue
		}
		if _, ok := p.Nav.Route(end, p.goalPoint); !ok {
			continue
		}
		return x, y, true
	}
	return 0, 0, false
}
