package learningenv

import (
	"fmt"
	"math"
	"q2coopbot/internal/quake"
)

type NavigationRewardEvidence struct {
	Available       bool    `json:"available"`
	Reason          string  `json:"reason,omitempty"`
	Goal            string  `json:"goal,omitempty"`
	Reference       string  `json:"reference,omitempty"`
	Before          float64 `json:"distance_before,omitempty"`
	After           float64 `json:"distance_after,omitempty"`
	BeforePotential float64 `json:"potential_before,omitempty"`
	AfterPotential  float64 `json:"potential_after,omitempty"`
	Scope           string  `json:"scope"`
}

// Fix the anchor from the actor's current observation for this transition.
// A moving partner/waypoint cannot earn progress for a stationary actor.
// Positive bounded potential gives nonpositive stationary and discounted
// closed-loop reward for a fixed anchor. Changing route anchors are a local
// experimental heuristic, not a global policy-invariance guarantee.
func (c RewardConfig) navigationShaping(s *Step, terminal bool) (float64, *NavigationRewardEvidence, error) {
	e := &NavigationRewardEvidence{Scope: "Current observed anchor, own displacement only; no peer trace coordinates. Local route shaping; global optimality unproven."}
	skip := func(reason string) (float64, *NavigationRewardEvidence, error) { e.Reason = reason; return 0, e, nil }
	if s.Next == nil {
		return skip("no_next_observation")
	}
	n := s.Observation.Navigation
	if n == nil || n.GoalRelative == nil {
		return skip("no_active_observed_goal")
	}
	e.Goal = n.Goal
	if n.Status != "ready" && n.Status != "direct_clear" {
		return skip("navigation_unavailable")
	}
	comfort := 128.0
	switch n.Goal {
	case "follow_teammate", "cover_teammate":
		if s.Observation.Teammate == nil {
			return skip("partner_not_currently_observed")
		}
	case "search_last_seen", "probe_last_seen":
		if n.LastTeammateAgeFrames == nil || *n.LastTeammateAgeFrames < 0 || *n.LastTeammateAgeFrames > 200 {
			return skip("unknown_or_stale_search_reference")
		}
		comfort = 48
	default:
		return skip("unsupported_navigation_goal")
	}
	finite := func(p quake.Vec3) bool {
		for _, v := range p {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return false
			}
		}
		return true
	}
	if !finite(s.Observation.Position) || !finite(s.Next.Position) || !finite(*n.GoalRelative) {
		return 0, e, fmt.Errorf("nonfinite navigation position")
	}
	relative := *n.GoalRelative
	e.Reference = "goal"
	goalDistance := quake.Distance(quake.Vec3{}, relative)
	if n.WaypointRelative != nil && (comfort == 48 || goalDistance > comfort) {
		if !finite(*n.WaypointRelative) {
			return 0, e, fmt.Errorf("nonfinite navigation waypoint")
		}
		relative = *n.WaypointRelative
		comfort = 24
		e.Reference = "waypoint"
	}
	if quake.Distance(s.Observation.Position, s.Next.Position) > 128 {
		return skip("nonphysical_or_teleport_displacement")
	}
	anchor := s.Observation.Position
	for i := range anchor {
		anchor[i] += relative[i]
	}
	if !finite(anchor) {
		return 0, e, fmt.Errorf("navigation anchor overflow")
	}
	e.Before = quake.Distance(s.Observation.Position, anchor)
	e.After = quake.Distance(s.Next.Position, anchor)
	phi := func(distance float64) float64 {
		return c.NavigationPotential * (1 - math.Min(1, math.Max(0, distance-comfort)/1024))
	}
	e.BeforePotential = phi(e.Before)
	e.AfterPotential = phi(e.After)
	if terminal {
		e.AfterPotential = 0
	}
	e.Available = true
	return c.AimGamma*e.AfterPotential - e.BeforePotential, e, nil
}
