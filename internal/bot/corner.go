package bot

import (
	"math"

	"q2coopbot/internal/quake"
)

// regroupCornerStep makes one short, grounded sidestep around a blocked AAS
// walking corner. The next snapshot replans from the observed new position.
func (p *Planner) regroupCornerStep(s quake.Snapshot, target quake.Vec3) (float64, float64, bool) {
	if p.World.Goal != "regroup_after_respawn" || !s.OnGround || p.Nav == nil || p.World.Geometry == nil || !p.World.Geometry.HasCollision() ||
		len(p.World.Route) == 0 || p.World.Route[0].Kind != 2 {
		return 0, 0, false
	}
	// The base2 slope can leave the player hull supported beside the wall,
	// 79 units from its next walking reach. A northward step is clear but
	// the usual rotated directions clip the wall or lack floor support.
	base2Slope := s.Map == "base2" && p.World.Route[0].ToArea == 405
	maxDistance := 64.0
	if base2Slope {
		maxDistance = 96
	}
	if quake.Horizontal(s.Self, target) > maxDistance {
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
	directions := []quake.Vec3{}
	for _, turn := range []float64{math.Pi / 3, -math.Pi / 3, math.Pi / 2, -math.Pi / 2} {
		x, y := 16*(ux*math.Cos(turn)-uy*math.Sin(turn)), 16*(ux*math.Sin(turn)+uy*math.Cos(turn))
		directions = append(directions, quake.Vec3{x, y, 0})
	}
	if base2Slope {
		directions = append(directions, quake.Vec3{0, 16, 0}, quake.Vec3{16, 0, 0}, quake.Vec3{0, -16, 0}, quake.Vec3{-16, 0, 0})
	}
	for _, dir := range directions {
		x, y := dir[0], dir[1]
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

// A locally safe sidestep may reselect the previous AAS area and repeat
// forever. After observing the same small patch several times, aim briefly
// for a later waypoint. The ordinary per-tick ground and door guards still
// validate every movement command; the override expires even if blocked.
func (p *Planner) regroupCornerEscape(s quake.Snapshot) (quake.Vec3, bool) {
	if p.World.Goal != "regroup_after_respawn" || !s.OnGround || p.Nav == nil || p.World.Geometry == nil || len(p.World.Route) == 0 {
		p.cornerHistory = nil
		p.cornerEscapeUntil = 0
		return quake.Vec3{}, false
	}
	if p.cornerEscapeUntil >= s.Frame {
		if quake.Horizontal(s.Self, p.cornerEscapeStart) <= 48 && quake.Horizontal(s.Self, p.cornerEscapeTarget) > 16 {
			return p.cornerEscapeTarget, true
		}
		p.cornerEscapeUntil = 0
	}
	if len(p.cornerHistory) >= 12 {
		p.cornerHistory = p.cornerHistory[1:]
	}
	p.cornerHistory = append(p.cornerHistory, s.Self)
	repeats := 0
	for i := 0; i+3 < len(p.cornerHistory); i++ {
		if quake.Horizontal(p.cornerHistory[i], s.Self) <= 5 {
			repeats++
		}
	}
	if repeats < 3 {
		return quake.Vec3{}, false
	}
	for _, wp := range p.World.Route {
		if wp.Kind == 11 {
			break
		}
		if quake.Horizontal(s.Self, wp.Position) < 96 || math.Abs(s.Self[2]-wp.Position[2]) > 32 {
			continue
		}
		if !p.cornerEscapeApproachClear(s, wp.Position) {
			continue
		}
		p.cornerEscapeStart = s.Self
		p.cornerEscapeTarget = wp.Position
		p.cornerEscapeUntil = s.Frame + 12
		p.cornerHistory = nil
		return wp.Position, true
	}
	return quake.Vec3{}, false
}

// Check several short steps before committing to a bypass. A single clear
// step can lead straight into the next wall and waste the whole override.
func (p *Planner) cornerEscapeApproachClear(s quake.Snapshot, target quake.Vec3) bool {
	at := s.Self
	g := p.World.Geometry
	for i := 0; i < 3; i++ {
		dx, dy := target[0]-at[0], target[1]-at[1]
		distance := math.Hypot(dx, dy)
		if distance <= 16 {
			return true
		}
		if g.GroundMoveHazardStep(p.Nav, at, dx, dy, 16) != "" || g.DoorMoveHazard(s.Movers, at, dx, dy) != "" {
			return false
		}
		end := at
		end[0] += dx / distance * 16
		end[1] += dy / distance * 16
		if !g.PlayerMoveClear(end, end) {
			return false
		}
		for _, mover := range s.Movers {
			if !g.MoverHullClear(mover, at, end) {
				return false
			}
		}
		at = end
	}
	return true
}
