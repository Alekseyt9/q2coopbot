package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// A short arc is rechecked each tick. This establishes a local movement
// corridor, not a clear full circle or protection against incoming projectiles.
func circleStep(w World, preferred float64) (quake.Vec3, float64, bool) {
	s := w.Snapshot
	p := combatSpacing(s)
	if p == nil || p.NeedSpace || !s.OnGround || s.Ducked || s.Health <= 0 ||
		w.Goal != "reach_level_exit" && w.Goal != "follow_teammate" && w.Goal != "cover_teammate" {
		return quake.Vec3{}, 0, false
	}
	var enemy *quake.Object
	for i := range s.Enemies {
		if s.Enemies[i].ID == p.Target {
			enemy = &s.Enemies[i]
		}
	}
	if enemy == nil {
		return quake.Vec3{}, 0, false
	}
	dx, dy := s.Self[0]-enemy.Origin[0], s.Self[1]-enemy.Origin[1]
	radius := math.Hypot(dx, dy)
	if radius < p.Minimum || radius > p.PreferredMax {
		return quake.Vec3{}, 0, false
	}
	if preferred == 0 {
		preferred = 1
	}
	for _, sign := range []float64{preferred, -preferred} {
		angle := sign * 16 / radius
		next := s.Self
		next[0] = enemy.Origin[0] + dx*math.Cos(angle) - dy*math.Sin(angle)
		next[1] = enemy.Origin[1] + dx*math.Sin(angle) + dy*math.Cos(angle)
		if !coverWalkClear(w, s.Self, next) || s.Teammate != nil && quake.Distance(next, *s.Teammate) > combatLeash(p) {
			continue
		}
		safe := true
		for _, other := range s.Enemies {
			if other.ID != enemy.ID && quake.Distance(s.Self, other.Origin) < 650 && quake.Distance(next, other.Origin) < quake.Distance(s.Self, other.Origin)-2 {
				safe = false
			}
		}
		candidate := s
		candidate.Self = next
		if !safe || !w.Geometry.ClearShot(candidate.EyePoint(), enemy.AimPoint()) || w.Geometry.DoorShotBlocked(s.Movers, candidate.EyePoint(), enemy.AimPoint()) ||
			s.Weapon == "Blaster" && !coverBlasterClear(w, next, *enemy) {
			continue
		}
		return next, sign, true
	}
	return quake.Vec3{}, 0, false
}

func (p *Planner) combatCircle(cmd quake.UserCmd) quake.UserCmd {
	s := p.World.Snapshot
	profile := combatSpacing(s)
	p.World.Command.MoveSource = "combat_circle"
	p.World.Command.MoveLimitReason = "combat_circle_blocked"
	cmd.Forward, cmd.Side, cmd.Up = 0, 0, 0
	if profile == nil || p.jump != nil || p.elevator != nil || p.button != nil {
		return cmd
	}
	if profile.NeedSpace {
		// Do not stand still waiting for the next model result when an
		// approaching enemy invalidates the chosen orbit radius.
		return p.combatRetreat(cmd, profile)
	}
	preferred := p.circleDirection
	if p.circleTarget != profile.Target {
		preferred = 0
	}
	next, sign, ok := circleStep(p.World, preferred)
	if !ok {
		return cmd
	}
	p.circleTarget, p.circleDirection = profile.Target, sign
	p.World.Command.MoveLimitReason = ""
	return worldMove(cmd, s, next[0]-s.Self[0], next[1]-s.Self[1], 80, p.World.Command.AimSource == "enemy")
}
