package demodata

import (
	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
)

const VerticalSelectionVersion = "teacher_vertical_candidates_v1"

func selectVertical(s learningenv.Step, effects learningenv.ServerOutcome, c policy.Capture, mode string) Selection {
	r := Selection{Version: VerticalSelectionVersion, Quality: "rejected"}
	deny := func(reason string) Selection { r.Reason = reason; return r }
	if mode != "rules" || s.Owner != "rules" || s.Provider != "rules" || c.Provider != "rules" || c.Selection == nil || c.Selection.Mode != "rules" || c.Selection.Owner != "rules" || c.Selection.Fallback != "" {
		return deny("not_rules_teacher")
	}
	if c.TeacherPrimitive != "vertical_flat_v1" || c.LimitReason != "teacher_vertical_primitive" || c.MoveLimitReason != "" || len(s.Interventions) != 0 {
		return deny("not_explicit_vertical_exercise")
	}
	o, n := s.Observation, s.Next
	if o.Identity.Life != 1 || s.Terminal || s.Truncated || n == nil || !policy.SameLife(o.Identity, n.Identity) || n.Identity.Frame != o.Identity.Frame+1 || o.Health <= 0 || n.Health <= 0 || o.AgeMS < 0 || o.AgeMS > 300 || n.AgeMS < 0 || n.AgeMS > 300 {
		return deny("incomplete_or_dead_transition")
	}
	if s.Execution == nil || !s.Execution.Matched || !s.Execution.WindowExclusive || s.Native == nil || !effects.Available || effects.Version != "server_step_effects_v1" {
		return deny("unproven_native_transition")
	}
	if effects.Deaths != 0 || effects.ReceivedHealthDamage != 0 || effects.SelfHealthDamage != 0 || effects.TeammateHealthDamage != 0 || effects.MonsterHealthDamage != 0 {
		return deny("damage_during_primitive")
	}
	a := s.AppliedAction
	if a.Identity != o.Identity || a.Version != policy.ActionVersion || c.Applied != a || c.AppliedCommand != s.Command || a.Weapon != "" || a.Attack || a.Forward != 0 || a.Side != 0 || s.Command.Buttons != 0 || s.Command.Forward != 0 || s.Command.Side != 0 || o.Geometry == nil || n.Geometry == nil {
		return deny("primitive_command_mismatch")
	}
	if _, err := policy.Command(o, a, [3]int16{}); err != nil {
		return deny("action_outside_contract")
	}
	switch a.Vertical {
	case "jump":
		if !o.OnGround || o.Ducked || n.OnGround || n.Ducked || o.PreviousCommand.Up > 0 || n.Position[2]-o.Position[2] < 4 || n.Velocity[2] <= 0 || o.Geometry.UpDistance < 40 || s.Command.Up <= 0 {
			return deny("jump_not_observed")
		}
		r.Reason = "observed_ground_takeoff"
	case "crouch":
		if !o.OnGround || !n.OnGround || o.Ducked || !n.Ducked || s.Command.Up >= 0 {
			return deny("crouch_transition_not_observed")
		}
		r.Reason = "observed_crouch_entry"
	case "release":
		if s.Command.Up != 0 {
			return deny("release_command_mismatch")
		}
		if o.OnGround && n.OnGround && o.Ducked && !n.Ducked {
			r.Reason = "observed_stand_up"
		} else if !o.OnGround && !o.Ducked && !n.Ducked && o.PreviousCommand.Up > 0 {
			r.Reason = "observed_jump_button_release"
		} else {
			return deny("release_transition_not_observed")
		}
	default:
		return deny("unsupported_vertical_action")
	}
	r.Heads.Vertical = true
	r.Quality = "scripted primitive; verified pose change; unreviewed; no tactical quality claim"
	return r
}
