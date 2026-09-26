package bot

import (
	"q2coopbot/internal/quake"
)

// planWalkOff accepts only a short AAS walk-off reach with a BSP-verified
// floor throughout the descent corridor. It cannot authorize arbitrary cliffs.
func (p *Planner) planWalkOff() bool {
	s, g := p.World.Snapshot, p.World.Geometry
	if p.World.Goal != "follow_teammate" || !s.OnGround || s.Health <= 0 || p.Nav == nil || !g.HasCollision() || p.elevator != nil || p.button != nil {
		return false
	}
	r := p.World.Route
	if len(r) < 2 || r[0].Kind != 7 || r[1].Kind != 7 || r[0].ToArea != r[1].ToArea || quake.Horizontal(s.Self, r[0].Position) > 64 {
		return false
	}
	end := r[1].Position
	if dz := s.Self[2] - end[2]; dz < 24 || dz > 160 {
		return false
	}
	for _, offset := range []quake.Vec3{{}, {24, 0, 0}, {-24, 0, 0}, {0, 24, 0}, {0, -24, 0}} {
		landing := end
		landing[0] += offset[0]
		landing[1] += offset[1]
		landing[2] += 0.125
		if quake.Horizontal(s.Self, landing) > 96 || !g.PlayerMoveClear(landing, landing) {
			continue
		}
		safe := true
		for _, corner := range []quake.Vec3{{}, {12, 12, 0}, {12, -12, 0}, {-12, 12, 0}, {-12, -12, 0}} {
			at := landing
			at[0] += corner[0]
			at[1] += corner[1]
			if _, ok := g.GroundDrop(at, 4); !ok {
				safe = false
			}
		}
		// Both the horizontal entry and the vertical column must fit the hull.
		above := landing
		above[2] = s.Self[2]
		if !safe || !g.PlayerMoveClear(s.Self, above) || !g.PlayerMoveClear(above, landing) || g.DoorShotBlocked(s.Movers, s.Self, above) || g.DoorShotBlocked(s.Movers, above, landing) {
			continue
		}
		for i := 0; i <= 12; i++ {
			at := s.Self
			at[0] += (landing[0] - s.Self[0]) * float64(i) / 12
			at[1] += (landing[1] - s.Self[1]) * float64(i) / 12
			drop, ok := g.GroundDrop(at, 160)
			// Keep the probe just above the BSP/AAS floor rounding boundary.
			at[2] -= drop - 0.5
			area := p.Nav.AreaFor(at)
			if !ok || area <= 0 || p.Nav.Areas[area].Contents&6 != 0 {
				safe = false
				break
			}
		}
		if !safe {
			continue
		}
		if _, ok := p.Nav.Route(landing, p.goalPoint); !ok {
			continue
		}
		p.jump = &jumpFlight{from: s.Self, landing: landing, frame: s.Frame, speed: 80, phase: 2, drop: true}
		return true
	}
	return false
}
