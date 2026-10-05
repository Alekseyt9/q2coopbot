// Package demodata prepares auditable teacher candidates, not trained weights
// or a claim that every selected controller action is tactically optimal.
package demodata

import (
	"math"
	"strings"

	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
)

const SelectionVersion = "teacher_candidates_v1"
const ReleaseSelectionVersion = "teacher_candidates_v2"
const DatasetVersion = "combat_demonstrations_v1"

type Heads struct {
	Movement bool `json:"movement"`
	Aim      bool `json:"aim"`
	Attack   bool `json:"attack"`
	Vertical bool `json:"vertical"`
	Weapon   bool `json:"weapon"`
}

type Selection struct {
	Version                string  `json:"version"`
	Quality                string  `json:"quality"`
	Heads                  Heads   `json:"heads"`
	Reason                 string  `json:"reason"`
	HorizontalDisplacement float64 `json:"horizontal_displacement"`
}

func Select(s learningenv.Step, effects learningenv.ServerOutcome, capture policy.Capture, mode string) Selection {
	return selectTeacher(s, effects, capture, mode, SelectionVersion)
}

func SelectVersion(s learningenv.Step, effects learningenv.ServerOutcome, capture policy.Capture, mode, version string) Selection {
	if version == VerticalSelectionVersion {
		return selectVertical(s, effects, capture, mode)
	}
	if version != SelectionVersion && version != ReleaseSelectionVersion {
		return Selection{Version: version, Quality: "rejected", Reason: "unsupported_selection_version"}
	}
	return selectTeacher(s, effects, capture, mode, version)
}

func selectTeacher(s learningenv.Step, effects learningenv.ServerOutcome, capture policy.Capture, mode, version string) Selection {
	r := Selection{Version: version, Quality: "rejected"}
	deny := func(reason string) Selection { r.Reason = reason; return r }
	if mode != "rules" || s.Owner != "rules" || s.Provider != "rules" || capture.Provider != "rules" || capture.Selection == nil || capture.Selection.Mode != "rules" || capture.Selection.Owner != "rules" || capture.Selection.Fallback != "" {
		return deny("not_rules_teacher")
	}
	if s.Observation.Identity.Life != 1 {
		return deny("after_first_life")
	}
	if s.Terminal || s.Truncated || s.Next == nil || !policy.SameLife(s.Observation.Identity, s.Next.Identity) || s.Next.Identity.Frame != s.Observation.Identity.Frame+1 || s.Observation.Health <= 0 || s.Next.Health <= 0 || s.Observation.AgeMS < 0 || s.Observation.AgeMS > 300 || s.Next.AgeMS < 0 || s.Next.AgeMS > 300 {
		return deny("incomplete_or_dead_transition")
	}
	if s.Execution == nil || !s.Execution.Matched || !s.Execution.WindowExclusive || s.Native == nil || !effects.Available || effects.Version != "server_step_effects_v1" {
		return deny("unproven_native_transition")
	}
	if capture.Changed || capture.LimitReason != "" || capture.MoveLimitReason != "" || len(s.Interventions) > 0 {
		return deny("guard_or_command_correction")
	}
	if effects.Deaths != 0 || effects.SelfHealthDamage != 0 || effects.TeammateHealthDamage != 0 || effects.ReceivedHealthDamage != 0 {
		return deny("damage_or_death_cost")
	}
	if s.Observation.Geometry == nil || len(s.Observation.Enemies) == 0 && version == SelectionVersion {
		return deny("missing_combat_observation")
	}
	a := s.AppliedAction
	if a.Version != policy.ActionVersion || a.Identity != s.Observation.Identity || capture.Applied != a || capture.AppliedCommand != s.Command {
		return deny("applied_label_mismatch")
	}
	if _, err := policy.Command(s.Observation, a, [3]int16{}); err != nil {
		return deny("action_outside_contract")
	}
	if len(s.Observation.Enemies) == 0 {
		// Sparse, attack-only teacher convention. Do not label navigation, aim,
		// cooldown, a guard correction, or unobserved targets as useful actions.
		if len(s.Next.Enemies) != 0 || a.Attack || s.Action.Attack || capture.Proposed.Attack || s.Command.Buttons&1 != 0 || a.Weapon != "" || s.Observation.Weapon != s.Next.Weapon || effects.MonsterHealthDamage != 0 || s.Index%10 != 0 {
			return deny("not_supported_targetless_release")
		}
		for _, event := range effects.Events {
			if event.Attacker == s.Observation.Identity.Actor && event.Take > 0 {
				return deny("outgoing_effect_during_release")
			}
		}
		r.Heads.Attack = true
		r.Quality = "auto_candidate; unreviewed; rules no-observed-target release convention"
		r.Reason = "no_observed_target_attack_release"
		return r
	}
	dx, dy := s.Next.Position[0]-s.Observation.Position[0], s.Next.Position[1]-s.Observation.Position[1]
	r.HorizontalDisplacement = math.Hypot(dx, dy)
	// Ground movement is only a verified primitive. It is not proof of retreat,
	// cover quality or improvement of the overall battle objective.
	r.Heads.Movement = s.Observation.OnGround && s.Next.OnGround && !s.Observation.Ducked && !s.Next.Ducked && a.Vertical == "release" && (a.Forward != 0 || a.Side != 0) && r.HorizontalDisplacement >= 2
	weapon := strings.ToLower(s.Observation.Weapon)
	shotgun := weapon == "shotgun" || strings.Contains(weapon, "/v_shotg/")
	if shotgun && s.Next.Weapon == s.Observation.Weapon && a.Weapon == "" && a.Attack {
		for _, e := range effects.Events {
			if e.Mod != 2 || e.Attacker != s.Observation.Identity.Actor || e.Inflictor != e.Attacker || e.HealthBefore <= 0 || e.Take <= 0 || !strings.HasPrefix(e.TargetClass, "monster_") {
				continue
			}
			for _, enemy := range s.Observation.Enemies {
				if enemy.ID == e.Target {
					r.Heads.Aim = true
					r.Heads.Attack = true
				}
			}
		}
	}
	if !r.Heads.Movement && !r.Heads.Aim {
		return deny("no_supported_useful_component")
	}
	r.Quality = "auto_candidate; unreviewed; rules tactical assistance present"
	r.Reason = "verified_ground_movement_or_shotgun_hitscan_effect"
	return r
}
