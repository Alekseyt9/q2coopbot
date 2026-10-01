package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

type bridgeLink struct {
	model               int
	origin, entry, exit quake.Vec3
	crossing            bool
}

func routeLength(from quake.Vec3, route []quake.Waypoint, goal quake.Vec3) float64 {
	cost := 0.
	for _, wp := range route {
		cost += quake.Distance(from, wp.Position)
		from = wp.Position
	}
	return cost + quake.Distance(from, goal)
}
func (p *Planner) stationaryBridge(model int, origin quake.Vec3) bool {
	s, old := p.World.Snapshot, p.doorPrevious
	if old.Map != s.Map || old.Frame+1 != s.Frame || old.Frame <= 0 {
		return false
	}
	found := false
	for _, m := range s.Movers {
		if m.Model == model && m.Origin == origin {
			found = true
		}
	}
	if !found {
		return false
	}
	for _, m := range old.Movers {
		if m.Model == model && m.Origin == origin {
			return true
		}
	}
	return false
}
func (p *Planner) bridgeLinkCorridor(from, to quake.Vec3, live quake.Mover) bool {
	g := p.World.Geometry
	d := quake.Horizontal(from, to)
	if d > 384 || math.Abs(from[2]-to[2]) > 18 {
		return false
	}
	start, end := from, to
	start[2] = math.Max(from[2], to[2])
	end[2] = start[2]
	if !g.PlayerMoveClear(from, start) || !g.PlayerMoveClear(start, end) {
		return false
	}
	for _, m := range p.World.Snapshot.Movers {
		if !g.MoverHullClear(m, from, start) || !g.MoverHullClear(m, start, end) {
			return false
		}
	}
	steps := int(math.Ceil(d / 4))
	if steps < 1 {
		steps = 1
	}
	for i := 0; i <= steps; i++ {
		t := float64(i) / float64(steps)
		at := quake.Vec3{from[0] + (to[0]-from[0])*t, from[1] + (to[1]-from[1])*t, start[2]}
		if _, ok := g.GroundDrop(at, 18); ok {
			continue
		}
		if _, ok := g.MoverFooting(live, at, 18); !ok {
			return false
		}
	}
	if _, reason := g.DoorMoveBlockStep(p.World.Snapshot.Movers, start, to[0]-from[0], to[1]-from[1], d); reason != "" {
		return false
	}
	return true
}
func (p *Planner) deployedBridgeRoute() ([]quake.Waypoint, *bridgeLink, bool) {
	if p.Nav == nil || p.World.Goal != "follow_teammate" || p.World.Snapshot.Health <= 0 || !p.World.Snapshot.OnGround || !p.World.Geometry.HasCollision() {
		return nil, nil, false
	}
	g := p.World.Geometry
	s := p.World.Snapshot
	bestCost := math.Inf(1)
	var best []quake.Waypoint
	var task *bridgeLink
	for _, e := range g.Entities {
		if e.Class != "func_door" {
			continue
		}
		b, ok := g.Model(e.Model)
		if !ok || b.Max[2]-b.Min[2] > 32 || b.Max[0]-b.Min[0] < 64 || b.Max[1]-b.Min[1] < 64 {
			continue
		}
		for _, m := range s.Movers {
			if m.Model != e.Model || !p.stationaryBridge(m.Model, m.Origin) {
				continue
			}
			lo, hi := b.Min, b.Max
			for a := range lo {
				lo[a] += m.Origin[a]
				hi[a] += m.Origin[a]
			}
			for axis := 0; axis < 2; axis++ {
				mid := quake.Vec3{(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2] + 24 + 18}
				a, z := mid, mid
				a[axis] = lo[axis] - 24
				z[axis] = hi[axis] + 24
				drop, ok := g.GroundDrop(a, 36)
				if !ok {
					continue
				}
				a[2] -= drop - .5
				drop, ok = g.GroundDrop(z, 36)
				if !ok {
					continue
				}
				z[2] -= drop - .5
				if !p.bridgeLinkCorridor(a, z, m) {
					continue
				}
				// This must actually span missing static support, rather than invent
				// an unnecessary link over an ordinary floor or a lowered door lip.
				mid[2] = a[2]
				if _, ok := g.GroundDrop(mid, 18); ok {
					continue
				}
				for _, ends := range [][2]quake.Vec3{{a, z}, {z, a}} {
					first, ok := p.Nav.Route(s.Self, ends[0])
					if !ok {
						continue
					}
					last, ok := p.Nav.Route(ends[1], p.goalPoint)
					if !ok {
						continue
					}
					r := append(append([]quake.Waypoint{}, first...), quake.Waypoint{Position: ends[0], Kind: 2}, quake.Waypoint{Position: ends[1], Kind: 2})
					r = append(r, last...)
					cost := routeLength(s.Self, r, p.goalPoint)
					if cost < bestCost {
						bestCost = cost
						best = r
						task = &bridgeLink{model: m.Model, origin: m.Origin, entry: ends[0], exit: ends[1]}
					}
				}
			}
		}
	}
	return best, task, task != nil
}
func (p *Planner) mayLeaveElevatorForBridge() bool {
	if p.elevator == nil {
		return true
	}
	s := p.World.Snapshot
	b, ok := p.World.Geometry.Model(p.elevator.model)
	if !ok {
		return false
	}
	for _, m := range s.Movers {
		if m.Model != p.elevator.model || !p.stationaryBridge(m.Model, m.Origin) {
			continue
		}
		// Never abandon a moving lift or an elevated landing in response to a
		// shortcut. The active board reach identifies its original bottom stop.
		for _, wp := range p.route {
			if wp.Model == m.Model && wp.ElevatorPhase == "board" {
				return math.Abs(m.Origin[2]-(b.Origin[2]-float64(wp.Rise))) < 1
			}
		}
	}
	return false
}
func (p *Planner) bridgeLinkNeedsExit() bool {
	task := p.bridgeLink
	if task == nil || !task.crossing {
		return false
	}
	s := p.World.Snapshot
	dx, dy := task.exit[0]-task.entry[0], task.exit[1]-task.entry[1]
	beyond := (s.Self[0]-task.exit[0])*dx+(s.Self[1]-task.exit[1])*dy >= 0
	if s.OnGround && (beyond || quake.Horizontal(s.Self, task.exit) <= 10) {
		if _, ok := p.World.Geometry.GroundDrop(s.Self, 4); ok && p.World.Geometry.PlayerMoveClear(s.Self, s.Self) {
			// Yielding to the player can carry us past the exact exit waypoint.
			// Complete on the verified static bank before selecting cover/yield.
			p.bridgeLink = nil
			p.routeKnown = false
			return false
		}
	}
	return true
}
func (p *Planner) bridgeLinkCommand(cmd quake.UserCmd) (quake.UserCmd, bool) {
	task := p.bridgeLink
	if task == nil {
		return cmd, false
	}
	s := p.World.Snapshot
	if !p.stationaryBridge(task.model, task.origin) {
		p.bridgeLink = nil
		p.routeKnown = false
		p.World.Command.MoveLimitReason = "bridge_link_moved_or_hidden"
		return cmd, true
	}
	if quake.Horizontal(s.Self, task.exit) <= 10 {
		p.bridgeLink = nil
		p.routeKnown = false
		return cmd, false
	}
	if !task.crossing && quake.Horizontal(s.Self, task.entry) > 16 {
		return cmd, false
	}
	task.crossing = true
	live := quake.Mover{Model: task.model, Origin: task.origin}
	end := task.exit
	if !s.OnGround || !p.bridgeLinkCorridor(s.Self, end, live) {
		p.World.Command.MoveLimitReason = "bridge_link_unverified"
		return cmd, true
	}
	cmd.Up = 0
	cmd.Buttons = 0
	p.World.Command = CommandDecision{MoveSource: "bridge_link", AimSource: "route", Skill: "bridge_link", MoveLimitReason: "observed_bridge_corridor"}
	return worldMove(cmd, s, end[0]-s.Self[0], end[1]-s.Self[1], 120, false), true
}
