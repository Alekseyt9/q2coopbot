package bot

import (
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

// diagnosticCheckedGroundRequest tests only a model-selected ground command
// rejected as unsupported by the simpler predictor. It never restores attack,
// changes aim, chooses an alternative movement, or permits jumps/air control.
func (p *Planner) diagnosticCheckedGroundRequest(s quake.Snapshot, requested, guarded quake.UserCmd, safety []policy.Intervention) (quake.UserCmd, []policy.Intervention) {
	if !s.OnGround || s.Health <= 0 || requested.Up > 0 || requested.Forward == 0 && requested.Side == 0 ||
		guarded.Forward != 0 || guarded.Side != 0 || guarded.Up != 0 {
		return guarded, nil
	}
	unsupported := false
	for _, change := range safety {
		if change.Component == "movement" {
			if change.Reason != "unsupported_motion_guard" {
				return guarded, nil
			}
			unsupported = true
		}
	}
	g := p.World.Geometry
	if !unsupported || g == nil || !g.MovementComplete() {
		return guarded, nil
	}
	candidate := guarded
	candidate.Forward, candidate.Side, candidate.Up = requested.Forward, requested.Side, requested.Up
	prediction := predictPmoveGround(s, candidate, g)
	if prediction == nil || !p.checkedCombatMotionPaths(s, candidate, prediction) {
		return guarded, nil
	}
	return candidate, []policy.Intervention{{Component: "movement", Reason: "diagnostic_checked_ground_request"}}
}
