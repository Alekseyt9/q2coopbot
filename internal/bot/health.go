package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// Pickups use item origins, not the standing player's origin. The common
// health model rests nine units below the standing player's center.
func healthStand(p quake.Vec3) quake.Vec3 { p[2] += 9.125; return p }

func (p *Planner) healthAllowed(at quake.Vec3, frame int) bool {
	return frame >= p.healthBanned[at]
}

// Ordinary baseq2 health is capped at 100. Don't spend a large kit on a
// scratch, including when a strategic recovery request overrides following.
// Critical health takes priority; unknown models use a conservative 25 HP.
func usefulHealth(s quake.Snapshot, item quake.Object) bool {
	if s.Health < 45 {
		return true
	}
	if s.Health >= 100 {
		return false
	}
	if item.HealthAmount == 2 || item.HealthAmount == 100 {
		// Stimpack and mega health can exceed the ordinary maximum.
		return true
	}
	amount := item.HealthAmount
	if amount <= 0 {
		amount = 25
	}
	return 100-int(s.Health) >= amount
}

// Keep an acquired pickup until it disappears or its progress budget expires.
// Reapplying the 300-unit acquisition radius every frame can alternate a
// downstairs pickup and an upstairs route on opposite sides of one step.
func (p *Planner) healthGoal(s quake.Snapshot) (quake.Vec3, bool) {
	if p.testSetupHold {
		return quake.Vec3{}, false
	}
	if p.healthActive {
		for _, item := range s.Pickups {
			if item.Class == "item_health" && usefulHealth(s, item) && healthStand(item.Origin) == p.healthTarget && p.healthAllowed(item.Origin, s.Frame) {
				return p.healthTarget, true
			}
		}
	}
	if p.healthActive {
		for _, r := range p.resources {
			if r.State == "unknown" && r.Item.Class == "item_health" && healthStand(r.Item.Origin) == p.healthTarget && usefulHealth(s, r.Item) && p.healthAllowed(r.Item.Origin, s.Frame) {
				r.Attempted = true
				return p.healthTarget, true
			}
		}
	}
	localAlternative := false
	if p.Nav != nil && p.World.Geometry.HasCollision() {
		for _, item := range s.Pickups {
			if item.Class == "item_health" && usefulHealth(s, item) && p.healthAllowed(item.Origin, s.Frame) && quake.Horizontal(s.Self, item.Origin) < 300 && math.Abs(healthStand(item.Origin)[2]-s.Self[2]) <= 64 {
				if _, ok := p.resourceWalkingRoute(s.Self, healthStand(item.Origin)); ok {
					localAlternative = true
					break
				}
			}
		}
	}
	for _, item := range s.Pickups {
		if item.Class == "item_health" && usefulHealth(s, item) && quake.Horizontal(s.Self, item.Origin) < 300 && p.healthAllowed(item.Origin, s.Frame) {
			// A nearby kit far above us may actually require a long detour.
			// Keep existing stair/drop recovery; the walking-only pickup budget
			// must not reject those established health routes.
			if localAlternative && healthStand(item.Origin)[2]-s.Self[2] > 64 {
				if _, ok := p.resourceRoute(s.Self, healthStand(item.Origin)); !ok {
					continue
				}
			}
			return healthStand(item.Origin), true
		}
	}
	for _, item := range p.rememberedCandidates(s) {
		if item.Class == "item_health" && usefulHealth(s, item) && p.healthAllowed(item.Origin, s.Frame) {
			at := healthStand(item.Origin)
			p.markResourceVisit(at)
			return at, true
		}
	}
	return quake.Vec3{}, false
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
		p.healthStarted = s.Frame
		p.healthLast = s.Self
	}
	if quake.Distance(s.Self, p.healthLast) > 16 {
		p.healthAt = s.Frame
		p.healthLast = s.Self
	}
	if s.Frame-p.healthAt < 25 && s.Frame-p.healthStarted < 100 {
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
