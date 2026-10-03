package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

type coverCycle struct {
	mapName        string
	target         int
	hide, peek     quake.Vec3
	stage          string
	started, until int
}

// A local wall is cover only from the currently observed threats. Never use
// this for explosives: blocking a line does not establish splash protection.
func coverHidden(g *quake.MapInfo, at quake.Vec3, enemies []quake.Object) bool {
	for _, e := range enemies {
		if quake.Distance(at, e.Origin) > 650 {
			continue
		}
		from := e.AimPoint()
		for _, height := range []float64{0, 22, 30} {
			to := at
			to[2] += height
			if g.ClearShot(from, to) {
				return false
			}
		}
	}
	return true
}

func coverWalkClear(w World, from, to quake.Vec3) bool {
	g := w.Geometry
	if g == nil || !g.MovementComplete() || math.Abs(from[2]-to[2]) > 1 {
		return false
	}
	d := quake.Horizontal(from, to)
	if d > 64 {
		return false
	}
	if d < .1 {
		return true
	}
	at := from
	for travel := 0.; travel < d; travel += 8 {
		step := math.Min(8, d-travel)
		dx, dy := to[0]-from[0], to[1]-from[1]
		next := at
		next[0] += dx / d * step
		next[1] += dy / d * step
		if g.GroundMoveHazardStep(nil, at, dx, dy, step) != "" || g.LaserMoveHazard(at, next) {
			return false
		}
		if _, reason := g.DoorMoveBlockStep(w.Snapshot.Movers, at, dx, dy, step); reason != "" {
			return false
		}
		for _, m := range w.Snapshot.Movers {
			if !g.MoverHullClear(m, at, next) {
				return false
			}
		}
		if w.Snapshot.Teammate != nil && quake.Horizontal(next, *w.Snapshot.Teammate) < 48 {
			return false
		}
		at = next
	}
	return true
}

func plannedCover(w World) *coverCycle {
	s := w.Snapshot
	if !s.OnGround || s.Ducked || s.Health <= 0 || w.Geometry == nil || w.Goal != "reach_level_exit" && w.Goal != "cover_teammate" && w.Goal != "follow_teammate" {
		return nil
	}
	profile := combatSpacing(s)
	if profile == nil || profile.NeedSpace {
		return nil
	}
	for _, e := range s.Enemies {
		if e.Class != "monster_soldier_light" && e.Class != "monster_soldier" && e.Class != "monster_soldier_ss" && e.Class != "monster_infantry" {
			return nil
		}
	}
	var target *quake.Object
	for i := range s.Enemies {
		if s.Enemies[i].ID == profile.Target {
			target = &s.Enemies[i]
			break
		}
	}
	if target == nil || target.Class != "monster_soldier_light" && target.Class != "monster_soldier" && target.Class != "monster_soldier_ss" && target.Class != "monster_infantry" {
		return nil
	}
	dx, dy := s.Self[0]-target.Origin[0], s.Self[1]-target.Origin[1]
	d := math.Hypot(dx, dy)
	if d < 1 {
		return nil
	}
	for _, dir := range []quake.Vec3{{-dy / d, dx / d, 0}, {dy / d, -dx / d, 0}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}} {
		for r := 16.; r <= 64; r += 8 {
			at := s.Self
			at[0] += r * dir[0]
			at[1] += r * dir[1]
			if s.Teammate != nil && quake.Distance(at, *s.Teammate) > combatLeash(profile) {
				continue
			}
			if !coverWalkClear(w, s.Self, at) || !coverHidden(w.Geometry, at, s.Enemies) {
				continue
			}
			groupSafe := true
			for _, e := range s.Enemies {
				if e.ID != target.ID && quake.Distance(at, e.Origin) < quake.Distance(s.Self, e.Origin)-2 {
					groupSafe = false
				}
			}
			if groupSafe {
				return &coverCycle{mapName: s.Map, target: target.ID, hide: at, peek: s.Self, stage: "withdraw", started: s.Frame, until: s.Frame + 100}
			}
		}
	}
	return nil
}

func (p *Planner) combatCoverCommand(s quake.Snapshot, cmd quake.UserCmd, tactic string) (quake.UserCmd, bool) {
	if p.cover == nil && tactic == "cover" && s.Frame >= p.coverRetry && p.jump == nil && p.elevator == nil && p.button == nil {
		p.cover = plannedCover(p.World)
	}
	c := p.cover
	if c == nil {
		return cmd, false
	}
	if s.Map != c.mapName || s.Health <= 0 || !s.OnGround || s.Frame < c.started || s.Frame > c.until || p.jump != nil || p.elevator != nil || p.button != nil {
		p.cover = nil
		p.coverRetry = s.Frame + 40
		return cmd, false
	}
	var enemy *quake.Object
	for i := range s.Enemies {
		if s.Enemies[i].ID == c.target {
			enemy = &s.Enemies[i]
			break
		}
	}
	if enemy == nil && c.stage != "return" {
		c.stage = "return"
	}
	if enemy != nil && (c.stage == "peek" || c.stage == "fire") {
		minimum, _ := combatDistanceBand(s.Weapon, enemy.Class)
		if quake.Distance(c.peek, enemy.Origin) < minimum {
			c.stage = "return"
		}
	}
	if enemy != nil && !coverHidden(p.World.Geometry, c.hide, s.Enemies) {
		p.cover = nil
		p.coverRetry = s.Frame + 40
		return cmd, false
	}
	to := c.hide
	if c.stage == "peek" || c.stage == "fire" {
		to = c.peek
	}
	p.World.Command.Skill = "combat_cover_" + c.stage
	p.World.Command.MoveSource = p.World.Command.Skill
	if c.stage == "fire" {
		cmd.Forward, cmd.Side, cmd.Up = 0, 0, 0
		if s.Frame-c.started >= 3 || enemy == nil || enemy.ClearShot == nil || !*enemy.ClearShot {
			c.stage = "return"
		}
		return cmd, true
	}
	cmd.Buttons = 0
	if quake.Horizontal(s.Self, to) < 4 {
		cmd.Forward, cmd.Side, cmd.Up = 0, 0, 0
		if quake.Horizontal(s.SelfVelocity, quake.Vec3{}) > 10 {
			return cmd, true
		}
		switch c.stage {
		case "withdraw":
			c.stage = "wait"
			c.started = s.Frame
		case "wait":
			if s.Frame-c.started >= 2 {
				c.stage = "peek"
			}
		case "peek":
			c.stage = "fire"
			c.started = s.Frame
		case "return":
			p.cover = nil
			p.coverRetry = s.Frame + 60
		}
		return cmd, true
	}
	if !coverWalkClear(p.World, s.Self, to) {
		p.cover = nil
		p.coverRetry = s.Frame + 40
		return cmd, false
	}
	return worldMove(cmd, s, to[0]-s.Self[0], to[1]-s.Self[1], math.Min(80, quake.Horizontal(s.Self, to)*10), p.World.Command.AimSource == "enemy"), true
}
