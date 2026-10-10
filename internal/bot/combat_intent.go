package bot

import "strings"

// CombatIntent arbitrates campaign travel against current observed combat.
// It does not pursue hidden enemies or override resource/objective goals.
type CombatIntent struct {
	Action string `json:"action"`
	Reason string `json:"reason"`
	Target int    `json:"target,omitempty"`
}

func combatIntent(w World) *CombatIntent {
	s := w.Snapshot
	if w.Campaign == nil || s.Teammate != nil && !w.Campaign.Leader || s.Health <= 0 {
		return nil
	}
	if w.Goal == "recover_health" {
		return &CombatIntent{Action: "recover", Reason: "health_resource_goal"}
	}
	if w.Goal != "reach_level_exit" {
		return &CombatIntent{Action: "route", Reason: "objective_priority"}
	}
	if s.Ammo <= 0 && !strings.Contains(strings.ToLower(s.Weapon), "blast") {
		return &CombatIntent{Action: "route", Reason: "no_usable_ammo"}
	}
	rangeLimit := 650.0
	if isRailgun(s.Weapon) {
		rangeLimit = 1000
	}
	if c := combatSpacing(s); c != nil && c.Distance < rangeLimit {
		return &CombatIntent{Action: "engage", Reason: "visible_campaign_threat", Target: c.Target}
	}
	return &CombatIntent{Action: "route", Reason: "no_close_visible_threat"}
}
