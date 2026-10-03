package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// A small grounded search repairs a walking corner, not the global AAS graph.
// Preserve intermediate points so replanning does not immediately undo escape.
func (p *Planner) planCornerDetour(s quake.Snapshot) bool {
	if !s.OnGround || p.Nav == nil || p.World.Geometry == nil ||
		(p.World.Goal != "reach_level_exit" && p.World.Goal != "regroup_after_respawn") ||
		len(p.World.Route) < 2 || p.World.Route[0].Kind != 2 {
		return false
	}
	var goals []quake.Vec3
	for i, wp := range p.World.Route {
		if wp.Kind != 2 && wp.Kind != 7 {
			break
		}
		if (i > 0 || quake.Horizontal(s.Self, wp.Position) > 112) && quake.Horizontal(s.Self, wp.Position) >= 24 && quake.Horizontal(s.Self, wp.Position) <= 384 && math.Abs(wp.Position[2]-s.Self[2]) <= 18 {
			goals = append(goals, wp.Position)
		}
		if len(goals) >= 4 {
			break
		}
		if wp.Kind == 7 {
			break
		}
	}
	if len(goals) == 0 {
		return false
	}
	type node struct {
		point  quake.Vec3
		parent int
		x, y   int
	}
	nodes := []node{{point: s.Self, parent: -1}}
	seen := map[[2]int]bool{{0, 0}: true}
	directions := [][2]int{{1, 0}, {0, -1}, {0, 1}, {-1, 0}, {1, -1}, {1, 1}, {-1, -1}, {-1, 1}}
	for head := 0; head < len(nodes) && head < 625; head++ {
		n := nodes[head]
		reached := head > 0 && p.cornerWalkOffFrom(s, n.point)
		var connection []quake.Vec3
		for _, goal := range goals {
			if head > 0 && quake.Horizontal(n.point, goal) <= 10 && math.Abs(n.point[2]-goal[2]) <= 8 {
				reached = true
			}
			if !reached && head > 0 && quake.Horizontal(s.Self, goal) > 112 {
				if path, ok := p.cornerRouteConnection(s, n.point, goal); ok {
					connection, reached = path, true
					break
				}
			}
		}
		if reached {
			var reverse []quake.Vec3
			for i := head; nodes[i].parent >= 0; i = nodes[i].parent {
				reverse = append(reverse, nodes[i].point)
			}
			p.cornerDetour = nil
			for i := len(reverse) - 1; i >= 0; i-- {
				p.cornerDetour = append(p.cornerDetour, reverse[i])
			}
			p.cornerDetour = append(p.cornerDetour, connection...)
			p.cornerDetourGoal, p.cornerDetourUntil = p.goalPoint, s.Frame+100
			p.cornerDetourRoute = append([]quake.Waypoint(nil), p.World.Route...)
			return true
		}
		for _, dir := range directions {
			if head == 0 && dir[0] != 0 && dir[1] != 0 {
				continue
			}
			stride := 1
			if head == 0 {
				stride = 2
			}
			x, y := n.x+stride*dir[0], n.y+stride*dir[1]
			key := [2]int{x, y}
			if seen[key] || absInt(x) > 12 || absInt(y) > 12 {
				continue
			}
			end := quake.Vec3{s.Self[0] + float64(x)*8, s.Self[1] + float64(y)*8, n.point[2]}
			end, ok := p.cornerGroundStep(s, n.point, end)
			if !ok {
				continue
			}
			seen[key] = true
			nodes = append(nodes, node{point: end, parent: head, x: x, y: y})
		}
	}
	return false
}

// Keep the small corner search bounded. A distant AAS waypoint can be joined
// only by a fully checked ground corridor, never by extending the grid blindly.
func (p *Planner) cornerRouteConnection(s quake.Snapshot, from, to quake.Vec3) ([]quake.Vec3, bool) {
	distance := quake.Horizontal(from, to)
	if distance < 1 || distance > 384 || math.Abs(to[2]-from[2]) > 18 || !p.World.Geometry.PlayerMoveClear(from, to) {
		return nil, false
	}
	steps := int(math.Ceil(distance / 8))
	path := make([]quake.Vec3, 0, steps)
	at := from
	for i := 1; i <= steps; i++ {
		t := float64(i) / float64(steps)
		next := quake.Vec3{from[0] + (to[0]-from[0])*t, from[1] + (to[1]-from[1])*t, at[2]}
		next, ok := p.cornerGroundStep(s, at, next)
		if !ok {
			return nil, false
		}
		path = append(path, next)
		at = next
	}
	return path, math.Abs(at[2]-to[2]) <= 8
}

// A walk-off's AAS start may already lie beyond static floor support. Stop
// searching at supported ground where the existing drop controller can verify
// its entire descent; do not demand a grounded path to that airborne point.
func (p *Planner) cornerWalkOffFrom(s quake.Snapshot, at quake.Vec3) bool {
	previous, flight := p.World.Snapshot, p.jump
	s.Self = at
	s.SelfVelocity = quake.Vec3{}
	p.World.Snapshot = s
	ok := p.planWalkOffRoute(p.World.Route)
	p.World.Snapshot, p.jump = previous, flight
	return ok
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func (p *Planner) cornerGroundStep(s quake.Snapshot, from, to quake.Vec3) (quake.Vec3, bool) {
	g := p.World.Geometry
	dx, dy := to[0]-from[0], to[1]-from[1]
	distance := math.Hypot(dx, dy)
	hazard := g.GroundMoveHazardStep(p.Nav, from, dx, dy, distance)
	if hazard == "static_hull_blocked" {
		// PM_StepSlideMove can stop its upward trace at a low ceiling.
		// Its hull can also rest on an edge when the center ray misses floor.
		for _, rise := range []float64{4, 8, 12, 16} {
			up, end := from, to
			up[2] += rise
			end[2] = up[2]
			if !g.PlayerMoveClear(from, up) || !g.PlayerMoveClear(up, end) {
				continue
			}
			if _, ok := cornerFooting(g, end, rise+.5); ok {
				hazard = ""
				break
			}
		}
	}
	if hazard != "" {
		return to, false
	}
	raised := to
	raised[2] = from[2] + 18
	height, ok := cornerFooting(g, raised, 36)
	if !ok {
		return to, false
	}
	to[2] = height
	if math.Abs(to[2]-from[2]) > 18 || !g.PlayerMoveClear(to, to) {
		return to, false
	}
	if g.PlayerTouchesHazard(to) || g.LaserMoveHazard(from, to) {
		return to, false
	}
	if _, reason := g.DoorMoveBlockStep(s.Movers, from, dx, dy, distance); reason != "" {
		return to, false
	}
	for _, mover := range s.Movers {
		if !g.MoverHullEscapeClear(mover, from, to) {
			return to, false
		}
	}
	return to, true
}

func cornerFooting(g *quake.MapInfo, at quake.Vec3, maxDrop float64) (float64, bool) {
	height := math.Inf(-1)
	for _, offset := range []quake.Vec3{{}, {-15.875, -15.875, 0}, {-15.875, 15.875, 0}, {15.875, -15.875, 0}, {15.875, 15.875, 0}} {
		point := at
		point[0] += offset[0]
		point[1] += offset[1]
		if d, ok := g.GroundDrop(point, maxDrop); ok {
			height = math.Max(height, point[2]-d+.25)
		}
	}
	return height, !math.IsInf(height, -1)
}

func (p *Planner) cornerDetourCommand(s quake.Snapshot, cmd quake.UserCmd) (quake.UserCmd, bool) {
	if len(p.cornerDetour) == 0 {
		return cmd, false
	}
	if !s.OnGround || s.Frame > p.cornerDetourUntil || p.cornerDetourGoal != p.goalPoint || (p.World.Goal != "reach_level_exit" && p.World.Goal != "regroup_after_respawn") {
		p.cornerDetour = nil
		return cmd, false
	}
	for len(p.cornerDetour) > 0 && quake.Horizontal(s.Self, p.cornerDetour[0]) < 4 && math.Abs(s.Self[2]-p.cornerDetour[0][2]) <= 18 {
		p.cornerDetour = p.cornerDetour[1:]
	}
	if len(p.cornerDetour) == 0 {
		p.routeKnown = false
		if p.planWalkOffRoute(p.cornerDetourRoute) {
			return p.jumpCommand(cmd)
		}
		return cmd, false
	}
	to := p.cornerDetour[0]
	dx, dy := to[0]-s.Self[0], to[1]-s.Self[1]
	distance := math.Hypot(dx, dy)
	if distance > 32 {
		p.cornerDetour = nil
		return cmd, false
	}
	if _, ok := p.cornerGroundStep(s, s.Self, to); !ok {
		p.cornerDetour = nil
		p.World.Command.MoveSource = "none"
		p.World.Command.Skill = "route_corner_detour"
		p.World.Command.LimitReason = "detour_step_changed"
		p.World.Command.MovePoint = &to
		return cmd, true
	}
	speed := math.Min(80, distance*10)
	for _, mover := range s.Movers {
		if !p.World.Geometry.MoverHullClear(mover, s.Self, s.Self) {
			speed = math.Min(160, distance*10)
			break
		}
	}
	p.World.Command.MoveSource = "route_corner_detour"
	p.World.Command.MovePoint = &to
	p.World.Command.Skill = "route_corner_detour"
	p.World.Command.LimitReason = "verified_ground_detour"
	// Brake lateral momentum before a tight grid turn. Ground friction alone
	// can carry the hull out of the checked segment and undo the detour.
	ux, uy := dx/distance, dy/distance
	cross := math.Abs(s.SelfVelocity[0]*uy - s.SelfVelocity[1]*ux)
	if cross > 20 {
		p.World.Command.MoveSource = "none"
		p.World.Command.LimitReason = "detour_turn_braking"
		return quake.UserCmd{Yaw: cmd.Yaw}, true
	}
	return worldMove(cmd, s, dx, dy, speed, false), true
}
