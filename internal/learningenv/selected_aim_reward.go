package learningenv

import "q2coopbot/internal/policy"

// This is a bounded experimental preference over an executed transition, not
// a globally telescoping state potential: switches/unknown references are masked.
// It supplies neither target choice nor aim commands to the runtime policy.
type AimReferenceEvidence struct {
	Entity  int     `json:"entity"`
	Track   int     `json:"track"`
	Applied bool    `json:"applied"`
	Reason  string  `json:"reason"`
	Before  float64 `json:"before"`
	After   float64 `json:"after"`
}

func matchingTarget(o policy.Observation, entity, track int) (policy.Enemy, bool) {
	for _, e := range o.Enemies {
		if e.ID == entity && e.Track != nil && *e.Track == track {
			return e, true
		}
	}
	return policy.Enemy{}, false
}

func (c RewardConfig) selectedAimShaping(s *Step, terminal bool) (float64, *AimReferenceEvidence, error) {
	evidence := &AimReferenceEvidence{Entity: s.Action.TargetEntity, Track: s.Action.TargetTrack}
	mask := func(reason string) (float64, *AimReferenceEvidence, error) {
		evidence.Reason = reason
		return 0, evidence, nil
	}
	if s.Owner != "provider" || evidence.Entity <= 0 || evidence.Track <= 0 {
		return mask("no_provider_target")
	}
	p := s.Observation.PreviousTarget
	if p == nil || p.Entity != evidence.Entity || p.Track != evidence.Track ||
		!policy.SameLife(p.Identity, s.Observation.Identity) || p.Identity.Frame+1 != s.Observation.Identity.Frame {
		return mask("new_or_changed_target")
	}
	current, exists := matchingTarget(s.Observation, evidence.Entity, evidence.Track)
	if !exists {
		return mask("current_target_unavailable")
	}
	_, known, err := policy.ObservedAimDirection(s.Observation, current)
	if err != nil {
		return 0, evidence, err
	}
	if !known {
		return mask("current_target_unavailable")
	}
	beforeObservation := s.Observation
	beforeObservation.Enemies = []policy.Enemy{current}
	before, err := aimPotentialWithKick(beforeObservation, c.AimPotential, c.AimKickAngles)
	if err != nil {
		return 0, evidence, err
	}
	after := 0.0
	if !terminal {
		p = s.Next.PreviousTarget
		if p == nil || p.Entity != evidence.Entity || p.Track != evidence.Track || p.Identity != s.Observation.Identity {
			return mask("next_intent_unavailable")
		}
		next, exists := matchingTarget(*s.Next, evidence.Entity, evidence.Track)
		if !exists {
			return mask("next_target_unavailable")
		}
		_, known, err = policy.ObservedAimDirection(*s.Next, next)
		if err != nil {
			return 0, evidence, err
		}
		if !known {
			return mask("next_target_unavailable")
		}
		afterObservation := *s.Next
		afterObservation.Enemies = []policy.Enemy{next}
		after, err = aimPotentialWithKick(afterObservation, c.AimPotential, c.AimKickAngles)
		if err != nil {
			return 0, evidence, err
		}
	}
	evidence.Applied, evidence.Reason, evidence.Before, evidence.After = true, "stable_selected_target", before, after
	return c.AimGamma*after - before, evidence, nil
}
