package bot

import (
	"math"

	"q2coopbot/internal/quake"
)

// planShortWalkDown handles a small ledge where AAS calls the next reach a
// walk, but the conservative one-tick ground probe sees a 24+ unit drop.
// It requires a nearby static floor with full hull support and a route onward.
func (p *Planner) planShortWalkDown() bool {
	s, g := p.World.Snapshot, p.World.Geometry
	if (p.World.Goal != "follow_teammate" && p.World.Goal != "regroup_after_respawn") || !s.OnGround || s.Health <= 0 || p.Nav == nil || g == nil || !g.HasCollision() || p.elevator != nil || p.button != nil || len(p.World.Route) == 0 || p.World.Route[0].Kind != 2 {
		return false
	}
	target := p.World.Route[0].Position
	dx, dy := target[0]-s.Self[0], target[1]-s.Self[1]
	distance := math.Hypot(dx, dy)
	if distance < 24 {
		return false
	}
	for _, step := range []float64{40, 48, 56, 64} {
		probe := s.Self
		probe[0] += dx / distance * step
		probe[1] += dy / distance * step
		drop, ok := g.GroundDrop(probe, 40)
		if !ok || drop < 24 || drop > 40 {
			continue
		}
		landing := probe
		landing[2] -= drop - 0.25
		if !g.PlayerMoveClear(landing, landing) {
			continue
		}
		valid := true
		for _, offset := range []quake.Vec3{{}, {12, 12, 0}, {12, -12, 0}, {-12, 12, 0}, {-12, -12, 0}} {
			at := landing
			at[0] += offset[0]
			at[1] += offset[1]
			if floor, supported := g.GroundDrop(at, 4); !supported || floor > 4 {
				valid = false
				break
			}
		}
		above := landing
		above[2] = s.Self[2]
		if !valid || !g.PlayerMoveClear(s.Self, above) || !g.PlayerMoveClear(above, landing) || g.DoorShotBlocked(s.Movers, s.Self, above) || g.DoorShotBlocked(s.Movers, above, landing) {
			continue
		}
		for _, mover := range s.Movers {
			if !g.MoverHullClear(mover, s.Self, above) || !g.MoverHullClear(mover, above, landing) {
				valid = false
				break
			}
		}
		if !valid {
			continue
		}
		for i := 0; i <= 8; i++ {
			at := s.Self
			at[0] += (landing[0] - s.Self[0]) * float64(i) / 8
			at[1] += (landing[1] - s.Self[1]) * float64(i) / 8
			floor, supported := g.GroundDrop(at, 40)
			if !supported || floor > 40 {
				valid = false
				break
			}
		}
		if !valid {
			continue
		}
		if _, ok := p.Nav.Route(landing, p.goalPoint); !ok {
			continue
		}
		p.jump = &jumpFlight{from: s.Self, landing: landing, frame: s.Frame, speed: 60, phase: 2, drop: true}
		return true
	}
	return false
}

// planWalkOff accepts only a short AAS walk-off reach with a BSP-verified
// floor throughout the descent corridor. It cannot authorize arbitrary cliffs.
func (p *Planner) planWalkOff() bool {
	s, g := p.World.Snapshot, p.World.Geometry
	if (p.World.Goal != "follow_teammate" && p.World.Goal != "regroup_after_respawn") || !s.OnGround || s.Health <= 0 || p.Nav == nil || !g.HasCollision() || p.elevator != nil || p.button != nil {
		return false
	}
	r := p.World.Route
	for len(r) > 2 && r[0].Kind == 2 && quake.Horizontal(s.Self, r[0].Position) <= 64 {
		r = r[1:]
	}
	if len(r) < 2 || r[0].Kind != 7 || r[1].Kind != 7 || r[0].ToArea != r[1].ToArea || quake.Horizontal(s.Self, r[0].Position) > 64 {
		return false
	}
	end := r[1].Position
	if dz := s.Self[2] - end[2]; dz < 24 || dz > 240 || dz > 160 && s.Health < 40 {
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
			drop, ok := g.GroundDrop(at, 240)
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
		// Air steering cannot brake ground velocity immediately. Keep enough
		// margin for drift after crossing the landing point on a short drop.
		speed := 60.0
		if s.Self[2]-landing[2] > 160 {
			speed = 40
		}
		p.jump = &jumpFlight{from: s.Self, landing: landing, frame: s.Frame, speed: speed, phase: 2, drop: true}
		return true
	}
	return false
}
