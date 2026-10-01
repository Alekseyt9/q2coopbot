package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// Times are ranking estimates, not promises: charge elevator waiting and HP,
// so a quick ordinary route wins over an unnecessary damaging shortcut.
func dropRouteSeconds(from quake.Vec3, route []quake.Waypoint, goal quake.Vec3) float64 {
	seconds := 0.0
	for _, wp := range route {
		seconds += quake.Distance(from, wp.Position) / 200
		if wp.Kind == 11 {
			seconds += 2
		}
		from = wp.Position
	}
	return seconds + quake.Distance(from, goal)/200
}

func preferDrop(dropSeconds, bypassSeconds float64, damage int) bool {
	return dropSeconds+float64(damage)*.25+1 < bypassSeconds
}

// Consider nearby recorded AAS walk-off edges even when the chosen AAS route
// approaches them through a much longer detour. The same full BSP/mover/floor
// validation as a normal walk-off still authorizes every candidate.
func (p *Planner) planNearbyWalkOff() bool {
	s := p.World.Snapshot
	if !s.OnGround || s.Health <= 0 || p.Nav == nil || p.World.Geometry == nil || len(p.World.Route) == 0 || s.Self[2]-p.goalPoint[2] < 64 || p.elevator != nil || p.button != nil || (p.World.Goal != "follow_teammate" && p.World.Goal != "regroup_after_respawn" && p.World.Goal != "reach_level_exit") {
		return false
	}
	bypass := dropRouteSeconds(s.Self, p.World.Route, p.goalPoint)
	var best *jumpFlight
	bestScore := math.Inf(1)
	for _, edges := range p.Nav.Edges {
		for _, e := range edges {
			if e.Kind != 7 || quake.Horizontal(s.Self, e.Start) > 64 || math.Abs(s.Self[2]-e.Start[2]) > 18 || s.Self[2]-e.End[2] < 64 || s.Self[2]-e.End[2] > 320 {
				continue
			}
			route := []quake.Waypoint{{Position: e.Start, Kind: 7, ToArea: e.To}, {Position: e.End, Kind: 7, ToArea: e.To}}
			if !p.planWalkOffRoute(route) {
				continue
			}
			candidate := p.jump
			p.jump = nil
			onward, ok := p.Nav.Route(candidate.landing, p.goalPoint)
			if !ok {
				continue
			}
			seconds := quake.Horizontal(s.Self, candidate.landing)/candidate.speed + math.Sqrt(2*(s.Self[2]-candidate.landing[2])/float64(s.Gravity)) + .2 + dropRouteSeconds(candidate.landing, onward, p.goalPoint)
			score := seconds + float64(candidate.expectedDamage)*.25
			if preferDrop(seconds, bypass, candidate.expectedDamage) && score < bestScore {
				best, bestScore = candidate, score
			}
		}
	}
	if best == nil {
		return false
	}
	p.jump = best
	return true
}
