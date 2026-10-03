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
	if (p.World.Goal != "follow_teammate" && p.World.Goal != "regroup_after_respawn" && p.World.Goal != "reach_level_exit") || !s.OnGround || s.Health <= 0 || p.Nav == nil || g == nil || !g.HasCollision() || p.elevator != nil || p.button != nil || len(p.World.Route) == 0 || p.World.Route[0].Kind != 2 {
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
		area := p.Nav.AreaFor(landing)
		if area <= 0 || p.Nav.Areas[area].Contents&6 != 0 || !g.PlayerMoveClear(landing, landing) {
			continue
		}
		valid := true
		for _, offset := range []quake.Vec3{{}, {12, 12, 0}, {12, -12, 0}, {-12, 12, 0}, {-12, -12, 0}} {
			at := landing
			at[0] += offset[0]
			at[1] += offset[1]
			// A descending stair can support the front of the hull on the
			// next tread. Require a floor within one native step, rather than
			// requiring all four corners to share a perfectly flat landing.
			if floor, supported := g.GroundDrop(at, 18.5); !supported || floor > 18.5 {
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
	return p.planWalkOffRoute(p.World.Route)
}

func (p *Planner) planWalkOffRoute(r []quake.Waypoint) bool {
	s, g := p.World.Snapshot, p.World.Geometry
	if (p.World.Goal != "follow_teammate" && p.World.Goal != "regroup_after_respawn" && p.World.Goal != "reach_level_exit") || !s.OnGround || s.Health <= 0 || p.Nav == nil || !g.HasCollision() || p.elevator != nil || p.button != nil {
		return false
	}
	for len(r) > 2 && r[0].Kind == 2 && quake.Horizontal(s.Self, r[0].Position) <= 64 {
		r = r[1:]
	}
	if len(r) < 2 || r[0].Kind != 7 || r[1].Kind != 7 || r[0].ToArea != r[1].ToArea || quake.Horizontal(s.Self, r[0].Position) > 64 {
		return false
	}
	end := r[1].Position
	if dz := s.Self[2] - end[2]; dz < 24 || dz > 320 {
		return false
	}
	offsets := []quake.Vec3{{}, {24, 0, 0}, {-24, 0, 0}, {0, 24, 0}, {0, -24, 0}, {24, 24, 0}, {24, -24, 0}, {-24, 24, 0}, {-24, -24, 0}, {48, 0, 0}, {-48, 0, 0}, {0, 48, 0}, {0, -48, 0}}
	if quake.Horizontal(end, p.goalPoint) <= 64 && math.Abs(end[2]-p.goalPoint[2]) <= 40 {
		goalOffset := quake.Vec3{p.goalPoint[0] - end[0], p.goalPoint[1] - end[1], 0}
		preferGoal := false
		for _, mover := range s.Movers {
			above := p.goalPoint
			above[2] = s.Self[2]
			if p.stationaryBridge(mover.Model, mover.Origin) {
				if _, ok := g.MoverFooting(mover, above, 320); ok {
					preferGoal = true
				}
			}
		}
		if preferGoal {
			// AAS endpoints at a lift's rim can retain native ground contact
			// with the upper ledge. Prefer the checked goal inside the brush.
			offsets = append([]quake.Vec3{goalOffset}, offsets...)
		} else {
			offsets = append(offsets, goalOffset)
		}
	}
	for _, offset := range offsets {
		landing := end
		landing[0] += offset[0]
		landing[1] += offset[1]
		landing[2] += 0.125
		// AAS walk-off endpoints can be above the real floor (notably the
		// suspended base2 exit). Project to a nearby static landing instead
		// of treating an airborne reach endpoint as standing support.
		floorTolerance := 4.0
		if drop, ok := g.GroundDrop(landing, 40); ok && drop > 4 {
			floorTolerance = 18.5
			floorOrigin := landing[2] - drop + .25
			for _, corner := range []quake.Vec3{{12, 12, 0}, {12, -12, 0}, {-12, 12, 0}, {-12, -12, 0}} {
				point := landing
				point[0] += corner[0]
				point[1] += corner[1]
				if d, ok := g.GroundDrop(point, 40); ok {
					floorOrigin = math.Max(floorOrigin, point[2]-d+.25)
				}
			}
			landing[2] = floorOrigin
		}
		// AAS records the static floor beneath a lift. Project onto an
		// observed stationary brush when it supplies the actual landing.
		for _, mover := range s.Movers {
			if !p.stationaryBridge(mover.Model, mover.Origin) {
				continue
			}
			above := landing
			above[2] = s.Self[2]
			if drop, ok := g.MoverFooting(mover, above, 320); ok {
				landing[2] = math.Max(landing[2], above[2]-drop+.25)
			}
		}
		damage := estimatedDropDamage(s.Self[2]-landing[2], s.SelfVelocity[2], s.Gravity)
		if s.Self[2]-landing[2] < 24 || s.Self[2]-landing[2] > 320 || !affordableDrop(s.Health, damage) {
			continue
		}
		if s.Map == "base1" && r[1].ToArea == 1898 && len(r) > 2 {
			// This reach ends exactly on the upper platform's XY boundary.
			// Native pmove can stop 1/8 unit short as input rounds down. Put
			// the target slightly beyond the edge toward the onward route;
			// all hull, support, descent and route checks below use this point.
			dx, dy := r[2].Position[0]-end[0], r[2].Position[1]-end[1]
			if d := math.Hypot(dx, dy); d > 2 {
				landing[0] += 2 * dx / d
				landing[1] += 2 * dy / d
			}
		}
		if quake.Horizontal(s.Self, landing) > 96 || !g.PlayerMoveClear(landing, landing) {
			continue
		}
		safe := true
		for _, corner := range []quake.Vec3{{}, {12, 12, 0}, {12, -12, 0}, {-12, 12, 0}, {-12, -12, 0}} {
			at := landing
			at[0] += corner[0]
			at[1] += corner[1]
			if _, ok := g.GroundDrop(at, floorTolerance); !ok {
				supported := false
				for _, mover := range s.Movers {
					if p.stationaryBridge(mover.Model, mover.Origin) {
						if _, ok := g.MoverFooting(mover, at, floorTolerance); ok {
							supported = true
						}
					}
				}
				if !supported {
					safe = false
				}
			}
		}
		// Both the horizontal entry and the vertical column must fit the hull.
		above := landing
		above[2] = s.Self[2]
		if !safe || !g.PlayerMoveClear(s.Self, above) || !g.PlayerMoveClear(above, landing) || g.DoorShotBlocked(s.Movers, s.Self, above) || g.DoorShotBlocked(s.Movers, above, landing) {
			continue
		}
		for _, mover := range s.Movers {
			if !g.MoverHullClear(mover, s.Self, above) || !g.MoverHullClear(mover, above, landing) {
				safe = false
				break
			}
		}
		if !safe || g.LaserMoveHazard(s.Self, above) || g.LaserMoveHazard(above, landing) {
			continue
		}
		for i := 0; i <= 12; i++ {
			at := s.Self
			at[0] += (landing[0] - s.Self[0]) * float64(i) / 12
			at[1] += (landing[1] - s.Self[1]) * float64(i) / 12
			drop, ok := g.GroundDrop(at, 320)
			// Keep the probe just above the BSP/AAS floor rounding boundary.
			at[2] -= drop - 0.5
			area := p.Nav.AreaFor(at)
			if !ok || g.PlayerTouchesHazard(at) || area > 0 && p.Nav.Areas[area].Contents&6 != 0 {
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
		p.jump = &jumpFlight{from: s.Self, landing: landing, frame: s.Frame, speed: speed, phase: 2, drop: true, expectedDamage: damage}
		return true
	}
	return false
}
