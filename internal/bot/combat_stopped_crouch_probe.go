package bot

import (
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

// diagnosticStoppedCrouch only restores the policy's own negative Up after a
// complete horizontal stop. It is an explicitly enabled synchronous experiment, not
// an ordinary-coop setting or permission to move, jump, or ignore hazards.
func (p *Planner) diagnosticStoppedCrouch(s quake.Snapshot, requested, guarded quake.UserCmd) (quake.UserCmd, []policy.Intervention) {
	if !s.OnGround || s.Health <= 0 || requested.Up >= 0 || guarded.Up != 0 || guarded.Forward != 0 || guarded.Side != 0 {
		return guarded, nil
	}
	g := p.World.Geometry
	if g == nil || !g.MovementComplete() {
		return guarded, nil
	}
	candidate := guarded
	candidate.Up = requested.Up
	prediction := predictPmoveGround(s, candidate, g)
	if prediction == nil || !p.checkedCombatMotionPaths(s, candidate, prediction) {
		return guarded, nil
	}
	return candidate, []policy.Intervention{{Component: "vertical", Reason: "diagnostic_checked_stopped_crouch"}}
}
