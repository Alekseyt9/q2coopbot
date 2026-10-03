package bot

import (
	"fmt"
	"q2coopbot/internal/quake"
	"regexp"
	"strings"
)

type CampaignDecision struct {
	Objective       string           `json:"objective"`
	State           string           `json:"state"`
	Exit            *quake.MapExit   `json:"exit,omitempty"`
	Preparation     *ExitPreparation `json:"preparation,omitempty"`
	RouteIndex      int              `json:"route_index,omitempty"`
	CompletedLevels int              `json:"completed_levels,omitempty"`
}

// A route explicitly resolves forward exits, including maps with return exits.
// Revisits are permitted, but adjacent duplicates cannot represent a transition.
func validateCampaignRoute(route []string) error {
	if len(route) == 0 {
		return nil
	}
	if len(route) < 2 || len(route) > 64 {
		return fmt.Errorf("run.campaign_route requires 2..64 map names")
	}
	name := regexp.MustCompile(`^[a-zA-Z0-9_]{1,64}$`)
	for i, m := range route {
		if !name.MatchString(m) || i > 0 && route[i-1] == m {
			return fmt.Errorf("invalid run.campaign_route map or adjacent duplicate")
		}
	}
	return nil
}

// Before the exit, improve health using known useful kits. This is bounded
// by the ordinary route/progress budget, not an obligation to reach full HP.
func (p *Planner) preparingForExit(s quake.Snapshot) bool {
	return !p.exitPreparationDone() && p.Campaign && s.Teammate == nil && s.Health > 0 && s.Health < 75 &&
		p.World.Campaign != nil && p.World.Campaign.Exit != nil &&
		quake.Horizontal(s.Self, p.World.Campaign.Exit.Center) < 768
}

func (p *Planner) preparingSuppliesForExit(s quake.Snapshot) bool {
	if p.exitPreparationDone() || !p.Campaign || s.Teammate != nil || s.Health < 45 || p.World.Campaign == nil || p.World.Campaign.Exit == nil || quake.Horizontal(s.Self, p.World.Campaign.Exit.Center) >= 768 {
		return false
	}
	for _, enemy := range s.Enemies {
		if enemy.ClearShot != nil && *enemy.ClearShot {
			return false
		}
	}
	return true
}

// Ambiguous destinations require next_map; do not guess a return transition.
func (p *Planner) campaignGoal(s quake.Snapshot) (quake.Vec3, bool) {
	d := &CampaignDecision{Objective: "complete_level", State: "exit_unknown"}
	p.World.Campaign = d
	next := p.CampaignNextMap
	if len(p.CampaignRoute) > 0 {
		index := p.campaignRouteIndex
		if index < 0 || index >= len(p.CampaignRoute) {
			d.State = "invalid_campaign_progress"
			return quake.Vec3{}, false
		}
		if s.Map != p.CampaignRoute[index] {
			if index+1 >= len(p.CampaignRoute) || s.Map != p.CampaignRoute[index+1] || p.campaignMap != p.CampaignRoute[index] || p.campaignDestination != s.Map {
				d.State = "unexpected_map_change"
				return quake.Vec3{}, false
			}
			p.campaignRouteIndex++
			index++
			p.campaignMap, p.campaignDestination = "", ""
		}
		d.RouteIndex, d.CompletedLevels = index, index
		if index == len(p.CampaignRoute)-1 {
			d.State = "campaign_completed"
			return quake.Vec3{}, false
		}
		next = p.CampaignRoute[index+1]
	} else if p.campaignMap != "" && s.Map != p.campaignMap {
		d.State = "unexpected_map_change"
		if s.Map == p.campaignDestination {
			d.State = "level_completed"
		}
		return quake.Vec3{}, false
	}
	exits := p.World.Geometry.Exits()
	var selected *quake.MapExit
	destination := ""
	matching := 0
	for i := range exits {
		x := &exits[i]
		base := strings.SplitN(strings.TrimPrefix(x.Destination, "*"), "$", 2)[0]
		if next != "" && base != next {
			continue
		}
		if destination != "" && destination != base {
			d.State = "ambiguous_exits"
			return quake.Vec3{}, false
		}
		destination = base
		matching++
		if selected == nil || quake.Distance(s.Self, x.Center) < quake.Distance(s.Self, selected.Center) {
			selected = x
		}
	}
	if selected == nil {
		return quake.Vec3{}, false
	}
	// Multiple triggers may lead to the same map. Prefer a supported contact
	// over a closer trigger suspended above a lower room. Keep the existing
	// button/drop approach when no standing contact is available.
	var contact *quake.Vec3
	if len(p.CampaignRoute) > 0 && matching > 1 {
		best := 1e30
		for i := range exits {
			x := &exits[i]
			if strings.SplitN(strings.TrimPrefix(x.Destination, "*"), "$", 2)[0] != destination {
				continue
			}
			if at, ok := p.campaignExitContact(*x, s.Self); ok {
				cost := quake.Distance(s.Self, at)
				if cost < best {
					best, selected = cost, x
					point := at
					contact = &point
				}
			}
		}
	}
	d.Exit = selected
	d.State = "approach_exit"
	p.campaignMap = s.Map
	p.campaignDestination = destination
	p.updateExitPreparation(s)
	if contact != nil {
		return *contact, true
	}
	goal := selected.Center
	if s.Self[2]+32 >= selected.Min[2] && s.Self[2]-24 <= selected.Max[2] {
		goal[2] = s.Self[2]
	}
	// A trigger center may float above the floor. Route to a supported player
	// origin whose hull touches the trigger, rather than a nearer upper deck.
	if drop, ok := p.World.Geometry.GroundDrop(selected.Center, 256); ok {
		standing := selected.Center
		standing[2] -= drop
		standing[2] += 0.25
		if standing[2]+32 > selected.Min[2] && standing[2]-24 < selected.Max[2] && p.World.Geometry.PlayerMoveClear(standing, standing) {
			goal = standing
		}
	}
	return goal, true
}

func (p *Planner) campaignExitContact(exit quake.MapExit, from quake.Vec3) (quake.Vec3, bool) {
	g := p.World.Geometry
	if g == nil || !g.HasCollision() || p.Nav == nil {
		return quake.Vec3{}, false
	}
	best, found := 1e30, false
	var chosen quake.Vec3
	for _, dx := range []float64{0, -8, 8, -16, 16} {
		for _, dy := range []float64{0, -8, 8, -16, 16} {
			at := exit.Center
			at[0] += dx
			at[1] += dy
			at[2] = exit.Max[2] + 24
			drop, ok := g.GroundDrop(at, 320)
			if !ok {
				continue
			}
			at[2] -= drop
			at[2] += 0.25
			if at[0]+16 <= exit.Min[0] || at[0]-16 >= exit.Max[0] || at[1]+16 <= exit.Min[1] || at[1]-16 >= exit.Max[1] || at[2]+32 <= exit.Min[2] || at[2]-24 >= exit.Max[2] || !g.PlayerMoveClear(at, at) || p.Nav.AreaFor(at) < 0 {
				continue
			}
			cost := quake.Distance(from, at)
			if cost < best {
				best, chosen, found = cost, at, true
			}
		}
	}
	return chosen, found
}
