package demodata

import (
	"q2coopbot/internal/aimquery"
	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
	"strings"
)

const AimQuerySelectionVersion = "counterfactual_observed_bbox_aim_v1"

// SelectAimQuery annotates an already verified learned-state observation.
// The query is not the executed command and has no associated outcome label.
func SelectAimQuery(s learningenv.Step, c policy.Capture, mode string) (Selection, *aimquery.Label, error) {
	r := Selection{Version: AimQuerySelectionVersion, Quality: "rejected", Reason: "not_learned_query_state"}
	if mode != "learned" || s.Owner != "provider" || !strings.HasPrefix(s.Provider, "ppo:") || c.Provider != s.Provider || c.Selection == nil || c.Selection.Mode != "learned" || c.Selection.Owner != "provider" || c.Selection.Fallback != "" {
		return r, nil, nil
	}
	if s.Native == nil || s.Execution == nil || !s.Execution.Matched || s.Next != nil && !s.Execution.WindowExclusive {
		r.Reason = "unproven_native_query_state"
		return r, nil, nil
	}
	if s.Observation.Identity.Life != 1 || s.Observation.Health <= 0 || s.Observation.AgeMS < 0 || s.Observation.AgeMS > 300 {
		r.Reason = "dead_or_stale_query_state"
		return r, nil, nil
	}
	q, err := aimquery.Query(s.Observation)
	if err != nil {
		return r, nil, err
	}
	if q == nil {
		r.Reason = "no_supported_observed_bbox_target"
		return r, nil, nil
	}
	r.Heads.Aim = true
	r.Quality = "counterfactual nominal aim; unexecuted; unreviewed; not hit or tactical proof"
	r.Reason = "nearest_observed_bbox_nominal_aim"
	return r, q, nil
}
