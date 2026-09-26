package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// AAS describes the space around a lift at rest, not the floor currently
// carrying the player. Join a verified same-height landing before routing.
func (p *Planner) platformExitCommand(cmd quake.UserCmd) (quake.UserCmd, bool) {
	s, g := p.World.Snapshot, p.World.Geometry
	if !s.OnGround || s.Health <= 0 || p.World.Goal != "follow_teammate" || p.Nav == nil || !g.HasCollision() || p.elevator != nil || p.button != nil {
		return cmd, false
	}
	if len(p.World.Route) > 0 && math.Abs(p.World.Route[0].Position[2]-s.Self[2]) <= 64 {
		return cmd, false
	}
	for _, entity := range g.Entities {
		if entity.Class != "func_plat" {
			continue
		}
		model, ok := g.Model(entity.Model)
		if !ok {
			continue
		}
		for _, mover := range s.Movers {
			if mover.Model != entity.Model {
				continue
			}
			lo, hi := model.Min, model.Max
			for axis := range lo {
				lo[axis] += mover.Origin[axis]
				hi[axis] += mover.Origin[axis]
			}
			if math.Abs(s.Self[2]-24-hi[2]) > 2 || s.Self[0] < lo[0] || s.Self[0] > hi[0] || s.Self[1] < lo[1] || s.Self[1] > hi[1] {
				continue
			}
			for _, target := range []quake.Vec3{{hi[0] + 24, s.Self[1], s.Self[2]}, {lo[0] - 24, s.Self[1], s.Self[2]}, {s.Self[0], hi[1] + 24, s.Self[2]}, {s.Self[0], lo[1] - 24, s.Self[2]}} {
				if quake.Horizontal(s.Self, target) > 160 || !g.PlayerMoveClear(s.Self, target) || g.DoorShotBlocked(s.Movers, s.Self, target) {
					continue
				}
				if _, ok := g.GroundDrop(target, 4); !ok {
					continue
				}
				area := p.Nav.ExactAreaFor(target)
				if area <= 0 || p.Nav.Areas[area].Contents&6 != 0 {
					continue
				}
				route, ok := p.Nav.Route(target, p.goalPoint)
				if !ok || len(route) > 0 && math.Abs(route[0].Position[2]-target[2]) > 32 {
					continue
				}
				safe := true
				for i := 1; i <= 20; i++ {
					at := s.Self
					at[0] += (target[0] - s.Self[0]) * float64(i) / 20
					at[1] += (target[1] - s.Self[1]) * float64(i) / 20
					// The 32-unit player hull still has at least eight units of
					// overlap while its centre crosses the small lift/floor seam.
					if at[0] >= lo[0]-8 && at[0] <= hi[0]+8 && at[1] >= lo[1]-8 && at[1] <= hi[1]+8 {
						continue
					}
					if _, ok := g.GroundDrop(at, 4); !ok {
						safe = false
						break
					}
				}
				if !safe {
					continue
				}
				p.routeKnown = false
				p.World.Command = CommandDecision{MoveSource: "platform_exit", AimSource: "route", Skill: "platform_exit", LimitReason: "platform_exit"}
				return worldMove(cmd, s, target[0]-s.Self[0], target[1]-s.Self[1], 120, false), true
			}
		}
	}
	return cmd, false
}
