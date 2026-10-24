package learningenv

import "math"

// This costs a rejected proposal, not standing still or the engine's collision
// response. Keep airborne, jumping and missing-ground safety decisions outside
// this experiment. The static collision guard and the executed command remain
// unchanged; native state is not added to the policy observation.
func (c RewardConfig) blockedGroundRequestCost(s *Step) float64 {
	if !s.Observation.OnGround || s.Action.Vertical != "release" ||
		math.Hypot(s.Action.Forward, s.Action.Side) < .25 ||
		math.Hypot(s.AppliedAction.Forward, s.AppliedAction.Side) >= .1 {
		return 0
	}
	for _, intervention := range s.Interventions {
		if intervention.Component == "movement" &&
			(intervention.Reason == "static_hull_blocked" || intervention.Reason == "static_hull_component_clipped") {
			return c.BlockedGroundRequest
		}
	}
	return 0
}
