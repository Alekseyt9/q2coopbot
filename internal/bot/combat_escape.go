package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

type cornerEscape struct {
	lastSelf               quake.Vec3
	lastFrame, stalled     int
	mapName                string
	target, started, until int
	path                   []quake.Vec3
	threatFloors           map[int]float64
}

// Check the whole edge, including a monster between two grid points.
func cornerThreatClear(s quake.Snapshot, from, to quake.Vec3, target int, minimum float64) bool {
	steps := max(1, int(math.Ceil(quake.Horizontal(from, to)/8)))
	for step := 1; step <= steps; step++ {
		at := from
		for axis := range at {
			at[axis] += (to[axis] - from[axis]) * float64(step) / float64(steps)
		}
		for _, e := range s.Enemies {
			if e.ID == target {
				if quake.Horizontal(at, e.Origin) < minimum {
					return false
				}
			} else if quake.Distance(s.Self, e.Origin) < 650 && quake.Distance(at, e.Origin) < quake.Distance(s.Self, e.Origin)-2 {
				return false
			}
		}
	}
	return true
}

func (p *Planner) continueCornerEscape(cmd quake.UserCmd, profile *CombatSpacing, enemy quake.Object) (quake.UserCmd, bool) {
	c, s := p.cornerEscape, p.World.Snapshot
	if c == nil {
		return cmd, false
	}
	if c.threatFloors != nil && c.target != enemy.ID {
		found := false
		for _, observed := range s.Enemies {
			if observed.ID == c.target && observed.Class == "monster_parasite" && observed.ClearShot != nil && *observed.ClearShot {
				enemy = observed
				found = true
				break
			}
		}
		if !found {
			p.cornerEscape = nil
			return cmd, false
		}
	}
	if !s.OnGround || s.Map != c.mapName || s.Frame < c.started || s.Frame > c.until || s.Health <= 0 || c.target != enemy.ID {
		p.cornerEscape = nil
		return cmd, false
	}
	if c.threatFloors != nil && s.Frame != c.lastFrame {
		if c.lastFrame != 0 && s.Frame == c.lastFrame+1 && quake.Horizontal(s.Self, c.lastSelf) < 1 {
			c.stalled++
		} else {
			c.stalled = 0
		}
		c.lastFrame = s.Frame
		c.lastSelf = s.Self
		if c.stalled >= 2 {
			p.cornerEscape = nil
			p.cornerDetourRetry = s.Frame + 20
			p.World.Command.MoveLimitReason = "combat_corner_detour_stalled"
			return cmd, false
		}
	}
	for len(c.path) > 0 && quake.Horizontal(s.Self, c.path[0]) < 8 {
		c.path = c.path[1:]
	}
	if len(c.path) == 0 {
		p.cornerEscape = nil
		return cmd, false
	}
	to := c.path[0]
	d := quake.Horizontal(s.Self, to)
	if d > 48 || !coverWalkClear(p.World, s.Self, to) || !cornerDetourThreatClear(s, s.Self, to, enemy.ID, 32, c.threatFloors) || s.Teammate != nil && quake.Distance(to, *s.Teammate) > combatLeash(profile) {
		p.cornerEscape = nil
		return cmd, false
	}
	p.World.Command.MoveSource = "combat_corner_escape"
	if c.threatFloors != nil {
		p.World.Command.MoveSource = "combat_corner_detour"
	}
	p.World.Command.MovePoint = &to
	p.World.Command.MoveLimitReason = ""
	p.World.Command.RetreatUrgent = true
	// A native command lasts 100ms: its displacement fits the checked edge.
	return worldMove(cmd, s, (to[0]-s.Self[0])/d, (to[1]-s.Self[1])/d, math.Min(240, d/0.1), p.World.Command.AimSource == "enemy"), true
}

// Only use this after ordinary retreat fails. A corner can require briefly
// getting closer to the tongue before reaching cover or a longer escape lane.
// Search a bounded supported walk, not a route through unseen monsters.
func (p *Planner) combatCornerEscape(cmd quake.UserCmd, profile *CombatSpacing, enemy quake.Object) quake.UserCmd {
	s := p.World.Snapshot
	if enemy.Class != "monster_parasite" || s.Health <= 0 || enemy.ClearShot == nil || !*enemy.ClearShot {
		return cmd
	}
	type cell struct{ x, y int }
	type node struct {
		cell
	}
	point := func(c cell) quake.Vec3 {
		return quake.Vec3{s.Self[0] + float64(c.x)*32, s.Self[1] + float64(c.y)*32, s.Self[2]}
	}
	distance := quake.Horizontal(s.Self, enemy.Origin)
	minimum := math.Max(32, distance-64)
	queue := []node{{}}
	seen := map[cell]bool{{}: true}
	parent := map[cell]cell{}
	for head := 0; head < len(queue); head++ {
		n := queue[head]
		at := point(n.cell)
		if head > 0 && (quake.Horizontal(at, enemy.Origin) >= distance+24 || coverHidden(p.World.Geometry, at, s.Enemies)) {
			var path []quake.Vec3
			for c := n.cell; c != (cell{}); c = parent[c] {
				path = append(path, point(c))
			}
			for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
				path[i], path[j] = path[j], path[i]
			}
			p.cornerEscape = &cornerEscape{mapName: s.Map, target: enemy.ID, started: s.Frame, until: s.Frame + 40, path: path}
			p.cornerUrgency = urgentRetreat{enemy.ID, s.Frame + 80}
			if result, ok := p.continueCornerEscape(cmd, profile, enemy); ok {
				return result
			}
			return cmd
		}
		for _, step := range []cell{{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}} {
			c := cell{n.x + step.x, n.y + step.y}
			if seen[c] || c.x*c.x+c.y*c.y > 64 {
				continue
			}
			next := point(c)
			if !cornerThreatClear(s, at, next, enemy.ID, minimum) || !coverWalkClear(p.World, at, next) {
				continue
			}
			if s.Teammate != nil && quake.Distance(next, *s.Teammate) > combatLeash(profile) {
				continue
			}
			seen[c] = true
			parent[c] = n.cell
			queue = append(queue, node{c})
		}
	}
	return p.combatMixedCornerDetour(cmd, profile, enemy)
}

// Fixed floors belong to the start of a detour, never a new budget per step.
func cornerDetourThreatClear(s quake.Snapshot, from, to quake.Vec3, target int, minimum float64, floors map[int]float64) bool {
	if floors == nil {
		return cornerThreatClear(s, from, to, target, minimum)
	}
	steps := max(1, int(math.Ceil(quake.Horizontal(from, to)/8)))
	for step := 1; step <= steps; step++ {
		at := from
		for axis := range at {
			at[axis] += (to[axis] - from[axis]) * float64(step) / float64(steps)
		}
		for _, e := range s.Enemies {
			floor, relaxed := floors[e.ID]
			if relaxed && (e.ID == target && e.Class == "monster_parasite" || e.Class == "monster_gunner") {
				if quake.Horizontal(at, e.Origin) < floor {
					return false
				}
			} else if quake.Distance(s.Self, e.Origin) < 650 && quake.Distance(at, e.Origin) < quake.Distance(s.Self, e.Origin)-2 {
				return false
			}
		}
	}
	return true
}

// A blocked mixed corner can require a longer walk past both observed threats.
// This is a bounded risk tradeoff, not splash protection or a safe-distance claim.
func (p *Planner) combatMixedCornerDetour(cmd quake.UserCmd, profile *CombatSpacing, enemy quake.Object) quake.UserCmd {
	s := p.World.Snapshot
	if s.Frame < p.cornerDetourRetry {
		return cmd
	}
	floors := map[int]float64{enemy.ID: math.Max(96, quake.Horizontal(s.Self, enemy.Origin)-128)}
	gunner := false
	for _, e := range s.Enemies {
		if e.Class == "monster_gunner" && e.ClearShot != nil && *e.ClearShot && quake.Distance(s.Self, e.Origin) <= 650 {
			floors[e.ID] = math.Max(64, quake.Horizontal(s.Self, e.Origin)-128)
			gunner = true
		}
	}
	if !gunner {
		return cmd
	}
	type cell struct{ x, y int }
	point := func(c cell) quake.Vec3 {
		return quake.Vec3{s.Self[0] + float64(c.x)*16, s.Self[1] + float64(c.y)*16, s.Self[2]}
	}
	queue := []cell{{}}
	seen := map[cell]bool{{}: true}
	parent := map[cell]cell{}
	distance := quake.Horizontal(s.Self, enemy.Origin)
	for head := 0; head < len(queue); head++ {
		c := queue[head]
		at := point(c)
		if head > 0 && (quake.Horizontal(at, enemy.Origin) >= distance+24 || coverHidden(p.World.Geometry, at, s.Enemies)) {
			var path []quake.Vec3
			for n := c; n != (cell{}); n = parent[n] {
				path = append(path, point(n))
			}
			for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
				path[i], path[j] = path[j], path[i]
			}
			if len(path) > 32 {
				continue
			}
			p.cornerEscape = &cornerEscape{mapName: s.Map, target: enemy.ID, started: s.Frame, until: s.Frame + 40, path: path, threatFloors: floors}
			p.cornerUrgency = urgentRetreat{enemy.ID, s.Frame + 80}
			if result, ok := p.continueCornerEscape(cmd, profile, enemy); ok {
				return result
			}
			return cmd
		}
		for _, d := range []cell{{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}} {
			n := cell{c.x + d.x, c.y + d.y}
			if seen[n] || n.x*n.x+n.y*n.y > 1024 {
				continue
			}
			next := point(n)
			if !cornerDetourThreatClear(s, at, next, enemy.ID, 96, floors) || !coverWalkClear(p.World, at, next) {
				continue
			}
			if s.Teammate != nil && quake.Distance(next, *s.Teammate) > combatLeash(profile) {
				continue
			}
			seen[n] = true
			parent[n] = c
			queue = append(queue, n)
		}
	}
	return cmd
}
