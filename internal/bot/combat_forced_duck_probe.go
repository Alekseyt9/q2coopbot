package bot

import (
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

// Opt-in diagnostic only. Restore the model's own release command when native
// CheckDuck retains the observed crouched pose. Never introduce a crouch action.
func (p *Planner) diagnosticForcedDuckRequest(s quake.Snapshot, requested, guarded quake.UserCmd, safety []policy.Intervention) (quake.UserCmd, []policy.Intervention) {
	if !s.OnGround || !s.Ducked || s.Health <= 0 || requested.Up != 0 || guarded.Up != 0 ||
		requested.Forward == 0 && requested.Side == 0 || guarded.Forward != 0 || guarded.Side != 0 {
		return guarded, nil
	}
	blocked := false
	for _, change := range safety {
		if change.Component == "movement" {
			if change.Reason != "static_hull_blocked" {
				return guarded, nil
			}
			blocked = true
		}
	}
	g := p.World.Geometry
	if !blocked || g == nil || !g.MovementComplete() {
		return guarded, nil
	}
	standing := g.TracePlayerWithEntities(s.Self, s.Self, false, s.Movers, s.CollisionBodies)
	ducked := g.TracePlayerWithEntities(s.Self, s.Self, true, s.Movers, s.CollisionBodies)
	if !standing.Valid || !standing.StartSolid && !standing.AllSolid || !ducked.Valid || ducked.StartSolid || ducked.AllSolid {
		return guarded, nil
	}
	candidate := guarded
	candidate.Forward, candidate.Side = requested.Forward, requested.Side
	prediction := predictPmoveGround(s, candidate, g)
	pose := candidate
	pose.Up = -400 // Path-hull selector only; candidate.Up remains zero.
	if prediction == nil || !p.checkedCombatMotionPaths(s, pose, prediction) {
		return guarded, nil
	}
	return candidate, []policy.Intervention{{Component: "movement", Reason: "diagnostic_forced_duck_request"}}
}
