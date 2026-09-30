package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// A same-map death does not invalidate the last visual observation. This
// target is a rendezvous point, never a fabricated visible player entity.
type respawnRegroup struct {
	entity int
	target quake.Vec3
}

func (p *Planner) observeRespawnRegroup(previous, s quake.Snapshot) {
	if previous.Frame > 0 && (previous.Map != s.Map || s.Frame < previous.Frame) {
		p.deathPoint = nil
		p.respawnRegroup = nil
		return
	}
	if previous.Frame > 0 && previous.Health > 0 && s.Health <= 0 {
		point := s.Self
		p.deathPoint = &point
	}
	if s.Teammate != nil {
		p.respawnRegroup = nil
		return
	}
	if p.respawnRegroup != nil && p.respawnRegroup.entity != 0 && (s.LastTeammate == nil || p.respawnRegroup.entity != s.LastTeammateEntity) {
		p.respawnRegroup = nil
	}
	if previous.Frame > 0 && previous.Health <= 0 && s.Health > 0 {
		if s.LastTeammate != nil && s.LastTeammateEntity > 0 {
			p.respawnRegroup = &respawnRegroup{entity: s.LastTeammateEntity, target: *s.LastTeammate}
		} else if p.deathPoint != nil {
			p.respawnRegroup = &respawnRegroup{target: *p.deathPoint}
		}
	}
}

func (p *Planner) respawnRegroupGoal(s quake.Snapshot) (quake.Vec3, bool) {
	r := p.respawnRegroup
	if r == nil || s.Health <= 0 || p.TestDisableSearch || p.testSetupHold || p.Nav == nil || p.World.GeometryStatus != "ready" {
		return quake.Vec3{}, false
	}
	if quake.Horizontal(s.Self, r.target) <= 64 && math.Abs(s.Self[2]-r.target[2]) <= 40 {
		p.respawnRegroup = nil
		return quake.Vec3{}, false
	}
	return r.target, true
}

// Some spawn areas have no outgoing AAS links. Join a nearby connected area
// only through a short, fully collision-checked walk on supported ground.
func (p *Planner) regroupEntryRoute(s quake.Snapshot, goal quake.Vec3) ([]quake.Waypoint, bool) {
	g := p.World.Geometry
	if !s.OnGround || g == nil || !g.HasCollision() || p.Nav == nil {
		return nil, false
	}
	if route, ok := p.regroupEntryRouteStep(s, goal, 16); ok {
		return route, true
	}
	// A coarse grid can skip the narrow strip connecting an isolated spawn.
	return p.regroupEntryRouteStep(s, goal, 4)
}

func (p *Planner) regroupEntryRouteStep(s quake.Snapshot, goal quake.Vec3, step float64) ([]quake.Waypoint, bool) {
	g := p.World.Geometry
	type node struct {
		at   quake.Vec3
		path []quake.Waypoint
	}
	queue := []node{{at: s.Self}}
	seen := map[quake.Vec3]bool{s.Self: true}
	for head := 0; head < len(queue); head++ {
		current := queue[head]
		if float64(len(current.path))*step >= 256 {
			continue
		}
		for _, dir := range []quake.Vec3{{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}} {
			at := current.at
			at[0] += dir[0] * step
			at[1] += dir[1] * step
			if seen[at] || math.Abs(at[0]-s.Self[0]) > 128 || math.Abs(at[1]-s.Self[1]) > 128 {
				continue
			}
			if !g.PlayerMoveClear(current.at, at) || !regroupGroundSupported(g, current.at, dir, step) {
				continue
			}
			if _, reason := g.DoorMoveBlockStep(s.Movers, current.at, dir[0], dir[1], step); reason != "" {
				continue
			}
			clear := true
			for _, mover := range s.Movers {
				if !g.MoverHullClear(mover, current.at, at) {
					clear = false
					break
				}
			}
			if !clear {
				continue
			}
			seen[at] = true
			path := append(append([]quake.Waypoint(nil), current.path...), quake.Waypoint{Position: at, Kind: 2})
			if route, ok := p.Nav.Route(at, goal); ok {
				return append(path, route...), true
			}
			queue = append(queue, node{at: at, path: path})
		}
	}
	return nil, false
}

// The collision floor can be slightly below the actual spawn support.
// Sample a short walk with at most one ordinary step down, not the two-unit
// flat-floor tolerance used by crouching passages. Command-time guards still
// check the observed floor, velocity, movers and hazards on every tick.
func regroupGroundSupported(g *quake.MapInfo, from, dir quake.Vec3, step float64) bool {
	for d := 0.0; d <= step; d += 2 {
		at := from
		at[0] += dir[0] * d
		at[1] += dir[1] * d
		if _, ok := g.GroundDrop(at, 18); !ok {
			return false
		}
	}
	return true
}
