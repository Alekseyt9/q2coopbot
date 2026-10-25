package learningenv

import (
	"fmt"

	"q2coopbot/internal/policy"
)

// MarkGoalBoundary labels an already supervisor-verified living transition.
// Verification of native kills is required separately; this never proves a win.
func MarkGoalBoundary(s *Step) error {
	if s.Next == nil || s.Terminal || s.Next.Health <= 0 || s.Next.Identity.Life != 1 ||
		!policy.SameLife(s.Observation.Identity, s.Next.Identity) ||
		s.Next.Identity.Frame != s.Observation.Identity.Frame+1 ||
		(s.Truncated && s.Reason != "control_handoff") {
		return fmt.Errorf("goal boundary lacks a complete living first-life transition")
	}
	if !s.Truncated {
		s.Terminal, s.Reason = true, "combat_goal_complete"
	} else {
		s.GoalHandoff = true
	}
	return nil
}

// MarkGoalHandoff uses an already independently verified supervisor receipt.
// It never labels visibility loss or disappearance as a completed goal.
func MarkGoalHandoff(s *Step, killFrame, observedFrame int) error {
	if !s.Truncated || s.Reason != "control_handoff" || s.Next == nil ||
		killFrame <= 0 || observedFrame <= killFrame ||
		s.Next.Identity.Frame < killFrame || s.Next.Identity.Frame > observedFrame {
		return fmt.Errorf("handoff outside verified goal interval")
	}
	return MarkGoalBoundary(s)
}

// A native objective may close long after the original kills (revival/boss phase).
// An earlier control handoff must not receive the eventual goal label.
func MarkGoalHandoffAtNativeObjective(s *Step, killFrame, observedFrame, nativeFrame int) error {
	if nativeFrame != observedFrame || s.Next == nil || s.Next.Identity.Frame != nativeFrame {
		return fmt.Errorf("handoff precedes verified native objective")
	}
	return MarkGoalHandoff(s, killFrame, observedFrame)
}

// GoalStop is supervisor evidence for an offline episode boundary, never a policy input.
type GoalStop struct {
	NativeObjective *CombatObjective `json:"native_objective,omitempty"`
	Version         string           `json:"version"`
	Reason          string           `json:"reason"`
	Spawncount      int              `json:"spawncount"`
	Actor           int              `json:"actor"`
	Classes         []string         `json:"classes"`
	KillFrame       int              `json:"kill_frame"`
	ObservedFrame   int              `json:"observed_frame"`
	Health          int              `json:"health"`
	Kills           []struct {
		Frame  int    `json:"frame"`
		Target int    `json:"target"`
		Class  string `json:"target_class"`
	} `json:"kills"`
}

func VerifyGoalStop(g GoalStop, release *CombatRelease, events []DamageEvent, observed policy.Observation, mixed bool) error {
	classes := []string{"monster_parasite"}
	if mixed {
		classes = append(classes, "monster_gunner")
	}
	return VerifyGoalStopForClasses(g, release, events, observed, classes)
}

// Expected classes come from the frozen fixture, never from the goal receipt.
func VerifyGoalStopForClasses(g GoalStop, release *CombatRelease, events []DamageEvent, observed policy.Observation, expectedClasses []string, maps ...string) error {
	fail := func() error { return fmt.Errorf("unverified combat goal stop") }
	expectedMap := "base1"
	if len(maps) > 1 {
		return fail()
	}
	if len(maps) == 1 {
		expectedMap = maps[0]
	}
	if expectedMap == "" {
		return fail()
	}
	id := observed.Identity
	if release == nil || g.Version != "combat_goal_stop_v1" || g.NativeObjective != nil || g.Reason != "combat_goal_complete" || id.Life != 1 || id.Connection != 1 || id.Map != expectedMap || id.Spawncount != release.Spawncount || g.Spawncount != id.Spawncount || g.Actor != id.Actor || g.ObservedFrame != id.Frame || g.Health != int(observed.Health) || g.Health <= 0 || g.KillFrame < release.Frame || g.ObservedFrame <= g.KillFrame {
		return fail()
	}
	expected := map[string]int{}
	for _, class := range expectedClasses {
		switch class {
		case "monster_parasite", "monster_gunner", "monster_soldier", "monster_soldier_light", "monster_soldier_ss", "monster_infantry", "monster_berserk", "monster_brain", "monster_chick", "monster_medic", "monster_mutant", "monster_gladiator", "monster_tank", "monster_tank_commander", "monster_supertank", "monster_jorg", "monster_makron", "monster_flyer", "monster_floater", "monster_hover", "monster_boss2", "monster_flipper":
		default:
			return fail()
		}
		expected[class]++
	}
	if len(expectedClasses) == 0 || len(expectedClasses) > 4 {
		return fail()
	}
	if len(g.Classes) != len(expectedClasses) || len(g.Kills) != len(expectedClasses) {
		return fail()
	}
	classes := map[string]int{}
	for _, c := range g.Classes {
		classes[c]++
		if expected[c] == 0 || classes[c] > expected[c] {
			return fail()
		}
	}
	killed := map[int]DamageEvent{}
	for _, e := range events {
		if e.Spawncount != id.Spawncount || e.Map != id.Map || e.Frame < release.Frame || e.Frame > id.Frame || e.HealthBefore <= 0 || e.HealthAfter > 0 {
			continue
		}
		if e.Target == id.Actor && e.TargetClass == "player" {
			return fail()
		}
		if e.Attacker == id.Actor && e.AttackerClass == "player" && e.Mod != 21 && expected[e.TargetClass] > 0 {
			if old, found := killed[e.Target]; !found || e.Frame < old.Frame {
				killed[e.Target] = e
			}
		}
	}
	if len(killed) < len(expectedClasses) {
		return fail()
	}
	last := 0
	targets := map[int]bool{}
	seen := map[string]int{}
	for _, k := range g.Kills {
		e, found := killed[k.Target]
		seen[k.Class]++
		if !found || seen[k.Class] > expected[k.Class] || targets[k.Target] || e.Frame != k.Frame || e.TargetClass != k.Class {
			return fail()
		}
		targets[k.Target] = true
		if k.Frame > last {
			last = k.Frame
		}
	}
	if last != g.KillFrame {
		return fail()
	}
	return nil
}
