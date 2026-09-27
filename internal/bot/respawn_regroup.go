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
	if s.Teammate != nil || s.LastTeammate == nil || s.LastTeammateEntity <= 0 || previous.Map != s.Map || s.Frame < previous.Frame {
		p.respawnRegroup = nil
		return
	}
	if p.respawnRegroup != nil && p.respawnRegroup.entity != s.LastTeammateEntity {
		p.respawnRegroup = nil
	}
	if p.testRespawnRegroup && previous.Frame > 0 && previous.Health <= 0 && s.Health > 0 {
		p.respawnRegroup = &respawnRegroup{entity: s.LastTeammateEntity, target: *s.LastTeammate}
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
	type node struct {
		at   quake.Vec3
		path []quake.Waypoint
	}
	queue := []node{{at: s.Self}}
	seen := map[quake.Vec3]bool{s.Self: true}
	for head := 0; head < len(queue); head++ {
		current := queue[head]
		if len(current.path) >= 16 {
			continue
		}
		for _, dir := range []quake.Vec3{{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}} {
			at := current.at
			at[0] += dir[0] * 16
			at[1] += dir[1] * 16
			if seen[at] || math.Abs(at[0]-s.Self[0]) > 128 || math.Abs(at[1]-s.Self[1]) > 128 {
				continue
			}
			if !g.PlayerMoveClear(current.at, at) || !g.CrouchStepClear(current.at, dir[0], dir[1], 16) {
				continue
			}
			if _, reason := g.DoorMoveBlockStep(s.Movers, current.at, dir[0], dir[1], 16); reason != "" {
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
