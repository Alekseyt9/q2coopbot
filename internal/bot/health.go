package bot

import "q2coopbot/internal/quake"

// Pickups use item origins, not the standing player's origin. The common
// health model rests nine units below the standing player's center.
func healthStand(p quake.Vec3) quake.Vec3 { p[2] += 9.125; return p }

func (p *Planner) healthAllowed(at quake.Vec3, frame int) bool {
	return frame >= p.healthBanned[at]
}

// A tempting but inaccessible pickup must not indefinitely own the follow
// goal. Budget progress in game frames and suppress that location briefly.
func (p *Planner) budgetHealthGoal(s quake.Snapshot, goal quake.Vec3) quake.Vec3 {
	if p.World.Goal != "recover_health" {
		p.healthActive = false
		return goal
	}
	if !p.healthActive || goal != p.healthTarget {
		p.healthActive = true
		p.healthTarget = goal
		p.healthAt = s.Frame
		p.healthLast = s.Self
	}
	if quake.Distance(s.Self, p.healthLast) > 16 {
		p.healthAt = s.Frame
		p.healthLast = s.Self
	}
	if s.Frame-p.healthAt < 25 {
		return goal
	}
	if p.healthBanned == nil {
		p.healthBanned = map[quake.Vec3]int{}
	}
	origin := goal
	origin[2] -= 9.125
	p.healthBanned[origin] = s.Frame + 150
	p.healthActive = false
	p.routeKnown = false
	p.World.Goal = "follow_teammate"
	if s.Teammate != nil {
		return *s.Teammate
	}
	return goal
}
