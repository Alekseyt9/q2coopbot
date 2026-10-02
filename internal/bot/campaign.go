package bot

import (
	"q2coopbot/internal/quake"
	"strings"
)

type CampaignDecision struct {
	Objective   string           `json:"objective"`
	State       string           `json:"state"`
	Exit        *quake.MapExit   `json:"exit,omitempty"`
	Preparation *ExitPreparation `json:"preparation,omitempty"`
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
	if p.campaignMap != "" && s.Map != p.campaignMap {
		d.State = "unexpected_map_change"
		if s.Map == p.campaignDestination {
			d.State = "level_completed"
		}
		return quake.Vec3{}, false
	}
	exits := p.World.Geometry.Exits()
	var selected *quake.MapExit
	destination := ""
	for i := range exits {
		x := &exits[i]
		base := strings.SplitN(strings.TrimPrefix(x.Destination, "*"), "$", 2)[0]
		if p.CampaignNextMap != "" && base != p.CampaignNextMap {
			continue
		}
		if destination != "" && destination != base {
			d.State = "ambiguous_exits"
			return quake.Vec3{}, false
		}
		destination = base
		if selected == nil || quake.Distance(s.Self, x.Center) < quake.Distance(s.Self, selected.Center) {
			selected = x
		}
	}
	if selected == nil {
		return quake.Vec3{}, false
	}
	d.Exit = selected
	d.State = "approach_exit"
	p.campaignMap = s.Map
	p.campaignDestination = destination
	p.updateExitPreparation(s)
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
