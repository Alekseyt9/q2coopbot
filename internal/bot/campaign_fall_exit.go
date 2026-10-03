package bot

import (
	"q2coopbot/internal/quake"
	"strings"
)

// A suspended exit can be crossed during an ordinary verified walk-off.
// Route to real ground below it; never invent an AAS node in mid-air.
func (p *Planner) campaignFallContact(exit quake.MapExit) (quake.Vec3, bool) {
	g := p.World.Geometry
	if g == nil || p.Nav == nil {
		return quake.Vec3{}, false
	}
	drop, ok := g.GroundDrop(exit.Center, 320)
	if !ok || drop < 32 {
		return quake.Vec3{}, false
	}
	landing := exit.Center
	landing[2] -= drop - .25
	area := p.Nav.ExactAreaFor(landing)
	if area <= 0 || p.Nav.Areas[area].Contents&6 != 0 || !g.PlayerMoveClear(exit.Center, landing) || !g.PlayerMoveClear(landing, landing) {
		return quake.Vec3{}, false
	}
	return landing, true
}

func (p *Planner) tryCampaignFallExit(s quake.Snapshot, destination string) bool {
	n := p.campaignDependencyNavigator(s)
	if n == nil {
		return false
	}
	for _, exit := range p.World.Geometry.Exits() {
		if strings.SplitN(strings.TrimPrefix(exit.Destination, "*"), "$", 2)[0] != destination || exit.Model == p.World.Campaign.Exit.Model {
			continue
		}
		landing, ok := p.campaignFallContact(exit)
		if !ok {
			continue
		}
		route, ok := n.Route(s.Self, landing)
		if !ok {
			continue
		}
		crosses := false
		for i := 0; i+1 < len(route); i++ {
			a, b := route[i], route[i+1]
			if a.Kind != 7 || b.Kind != 7 || a.ToArea != b.ToArea || a.Position[2] <= exit.Max[2]+24 || b.Position[2] >= exit.Min[2]-32 {
				continue
			}
			// Both XY endpoints must remain inside the trigger footprint.
			// This deliberately rejects glancing or diagonal crossings.
			inside := true
			for _, point := range []quake.Vec3{a.Position, b.Position} {
				inside = inside && point[0] >= exit.Min[0]+16 && point[0] <= exit.Max[0]-16 && point[1] >= exit.Min[1]+16 && point[1] <= exit.Max[1]-16
			}
			drop := a.Position[2] - b.Position[2]
			if inside && drop <= 320 && affordableDrop(s.Health, estimatedDropDamage(drop, 0, s.Gravity)) {
				crosses = true
				break
			}
		}
		if crosses {
			p.campaignExitOverride, p.campaignExitBlock = exit.Model, p.campaignDependency
			p.campaignDependency = nil
			p.routeKnown = false
			return true
		}
	}
	return false
}
