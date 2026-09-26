package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// bridgeSupport uses only the observed translating brush under the teammate.
func (p *Planner) bridgeSupport(goal quake.Vec3) (quake.Vec3, quake.Vec3, bool) {
	g := p.World.Geometry
	if !g.HasCollision() {
		return quake.Vec3{}, quake.Vec3{}, false
	}
	for _, e := range g.Entities {
		if e.Class != "func_door" {
			continue
		}
		m, ok := g.Model(e.Model)
		if !ok {
			continue
		}
		for _, live := range p.World.Snapshot.Movers {
			if live.Model != e.Model {
				continue
			}
			lo, hi := m.Min, m.Max
			for a := range lo {
				lo[a] += live.Origin[a]
				hi[a] += live.Origin[a]
			}
			if goal[0] >= lo[0]+16 && goal[0] <= hi[0]-16 && goal[1] >= lo[1]+16 && goal[1] <= hi[1]-16 && math.Abs(goal[2]-24-hi[2]) < 2 {
				return lo, hi, true
			}
		}
	}
	return quake.Vec3{}, quake.Vec3{}, false
}
func (p *Planner) bridgeCorridor(from, to, lo, hi quake.Vec3) bool {
	g := p.World.Geometry
	if quake.Horizontal(from, to) > 384 || math.Abs(from[2]-to[2]) > 18 {
		return false
	}
	above := to
	above[2] = from[2]
	if !g.PlayerMoveClear(from, above) || g.DoorShotBlocked(p.World.Snapshot.Movers, from, above) {
		return false
	}
	for i := 0; i <= 48; i++ {
		at := from
		at[0] += (to[0] - from[0]) * float64(i) / 48
		at[1] += (to[1] - from[1]) * float64(i) / 48
		if at[0] >= lo[0] && at[0] <= hi[0] && at[1] >= lo[1] && at[1] <= hi[1] && math.Abs(at[2]-24-hi[2]) <= 18 {
			continue
		}
		if _, ok := g.GroundDrop(at, 24); !ok {
			return false
		}
	}
	return true
}
func (p *Planner) bridgeRoute() ([]quake.Waypoint, bool) {
	if p.World.Goal != "follow_teammate" {
		return nil, false
	}
	lo, hi, ok := p.bridgeSupport(p.goalPoint)
	if !ok {
		return nil, false
	}
	goal := p.goalPoint
	var best []quake.Waypoint
	score := math.Inf(1)
	for _, at := range []quake.Vec3{{goal[0], lo[1] - 24, goal[2] + 18}, {goal[0], hi[1] + 24, goal[2] + 18}, {lo[0] - 24, goal[1], goal[2] + 18}, {hi[0] + 24, goal[1], goal[2] + 18}} {
		drop, ok := p.World.Geometry.GroundDrop(at, 24)
		if !ok {
			continue
		}
		at[2] -= drop - .5
		area := p.Nav.ExactAreaFor(at)
		if area <= 0 || p.Nav.Areas[area].Contents&6 != 0 || !p.bridgeCorridor(at, goal, lo, hi) {
			continue
		}
		route, ok := p.Nav.Route(p.World.Snapshot.Self, at)
		if !ok {
			continue
		}
		cost := 0.
		prev := p.World.Snapshot.Self
		for _, wp := range route {
			cost += quake.Distance(prev, wp.Position)
			prev = wp.Position
		}
		cost += quake.Distance(prev, at) + quake.Distance(at, goal)
		if cost < score {
			score = cost
			best = append(route, quake.Waypoint{Position: at, Kind: 2})
		}
	}
	return best, best != nil
}
func (p *Planner) bridgeCommand(cmd quake.UserCmd) (quake.UserCmd, bool) {
	s := p.World.Snapshot
	if !s.OnGround || p.World.Goal != "follow_teammate" || p.elevator != nil || p.button != nil {
		return cmd, false
	}
	lo, hi, ok := p.bridgeSupport(p.goalPoint)
	if !ok || !p.bridgeCorridor(s.Self, p.goalPoint, lo, hi) {
		return cmd, false
	}
	cmd.Buttons = 0
	p.World.Command = CommandDecision{MoveSource: "bridge", AimSource: "route", Skill: "bridge", LimitReason: "bridge_crossing"}
	return worldMove(cmd, s, p.goalPoint[0]-s.Self[0], p.goalPoint[1]-s.Self[1], 120, false), true
}

func (p *Planner) bridgeNeedsApproach(goal quake.Vec3) bool {
	lo, hi, ok := p.bridgeSupport(goal)
	if !ok {
		return false
	}
	s := p.World.Snapshot.Self
	return s[0] < lo[0]+8 || s[0] > hi[0]-8 || s[1] < lo[1]+8 || s[1] > hi[1]-8 || math.Abs(s[2]-goal[2]) > 18
}
