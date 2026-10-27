package bot

import "q2coopbot/internal/quake"

// RecordedMotionAudit is an offline diagnostic. It does not authorize commands
// or change the live guard, policy observations, weights, or action selection.
type RecordedMotionAudit struct {
	GroundSurface                    string                 `json:"ground_surface"`
	Applied                          *PmoveGroundPrediction `json:"applied_prediction"`
	AppliedPathChecked               bool                   `json:"applied_path_checked"`
	Proposed                         *PmoveGroundPrediction `json:"proposed_prediction"`
	ProposedPathChecked              bool                   `json:"proposed_path_checked"`
	ForcedDuckObserved               bool                   `json:"forced_duck_observed"`
	ProposedEffectivePosePathChecked bool                   `json:"proposed_effective_pose_path_checked"`
	StoppedCrouchEligible            bool                   `json:"stopped_crouch_eligible"`
	StoppedCrouchCommand             quake.UserCmd          `json:"stopped_crouch_command"`
	StoppedCrouch                    *PmoveGroundPrediction `json:"stopped_crouch_prediction"`
	StoppedCrouchPathChecked         bool                   `json:"stopped_crouch_path_checked"`
}

// DiagnoseRecordedCombatMotion reuses the existing observed-state physics and
// path checks. Passing these diagnostics alone is not a native execution proof.
func DiagnoseRecordedCombatMotion(s quake.Snapshot, proposed, applied quake.UserCmd, g *quake.MapInfo) RecordedMotionAudit {
	r := RecordedMotionAudit{}
	if g == nil || !g.MovementComplete() {
		return r
	}
	r.GroundSurface = g.GroundFrictionStatus(s.Self)
	p := Planner{World: World{Geometry: g, Snapshot: s}}
	r.Applied = predictPmoveGround(s, applied, g)
	if r.Applied != nil {
		r.AppliedPathChecked = p.checkedCombatMotionPaths(s, applied, r.Applied)
	}
	r.Proposed = predictPmoveGround(s, proposed, g)
	if r.Proposed != nil {
		r.ProposedPathChecked = p.checkedCombatMotionPaths(s, proposed, r.Proposed)
		// A release command does not stand up when native PM_CheckDuck finds
		// the standing hull blocked. Diagnose that existing pose explicitly;
		// this neither edits the command nor grants live movement permission.
		standing := g.TracePlayerWithEntities(s.Self, s.Self, false, s.Movers, s.CollisionBodies)
		ducked := g.TracePlayerWithEntities(s.Self, s.Self, true, s.Movers, s.CollisionBodies)
		r.ForcedDuckObserved = s.Ducked && proposed.Up == 0 && standing.Valid &&
			(standing.StartSolid || standing.AllSolid) && ducked.Valid && !ducked.StartSolid && !ducked.AllSolid
		poseCommand := proposed
		if r.ForcedDuckObserved {
			poseCommand.Up = -400 // Select only the existing path check's hull.
		}
		r.ProposedEffectivePosePathChecked = p.checkedCombatMotionPaths(s, poseCommand, r.Proposed)
	}
	r.StoppedCrouchEligible = s.OnGround && s.Health > 0 && proposed.Up < 0 && applied.Up == 0 && applied.Forward == 0 && applied.Side == 0
	if r.StoppedCrouchEligible {
		candidate := applied
		candidate.Up = proposed.Up
		r.StoppedCrouchCommand = candidate
		r.StoppedCrouch = predictPmoveGround(s, candidate, g)
		if r.StoppedCrouch != nil {
			r.StoppedCrouchPathChecked = p.checkedCombatMotionPaths(s, candidate, r.StoppedCrouch)
		}
	}
	return r
}
