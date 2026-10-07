package learningenv

import (
	"fmt"

	"q2coopbot/internal/policy"
)

// GoalStop is supervisor evidence for an offline episode boundary, never a policy input.
type GoalStop struct {
	Version       string   `json:"version"`
	Reason        string   `json:"reason"`
	Spawncount    int      `json:"spawncount"`
	Actor         int      `json:"actor"`
	Classes       []string `json:"classes"`
	KillFrame     int      `json:"kill_frame"`
	ObservedFrame int      `json:"observed_frame"`
	Health        int      `json:"health"`
	Kills         []struct {
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
func VerifyGoalStopForClasses(g GoalStop, release *CombatRelease, events []DamageEvent, observed policy.Observation, expectedClasses []string) error {
	fail := func() error { return fmt.Errorf("unverified combat goal stop") }
	id := observed.Identity
	if release == nil || g.Version != "combat_goal_stop_v1" || g.Reason != "combat_goal_complete" || id.Life != 1 || id.Connection != 1 || id.Map != "base1" || id.Spawncount != release.Spawncount || g.Spawncount != id.Spawncount || g.Actor != id.Actor || g.ObservedFrame != id.Frame || g.Health != int(observed.Health) || g.Health <= 0 || g.KillFrame <= release.Frame || g.ObservedFrame <= g.KillFrame {
		return fail()
	}
	expected := map[string]bool{}
	for _, class := range expectedClasses {
		switch class {
		case "monster_parasite", "monster_gunner", "monster_soldier", "monster_infantry":
		default:
			return fail()
		}
		if expected[class] {
			return fail()
		}
		expected[class] = true
	}
	if len(expected) == 0 || len(expected) > 2 {
		return fail()
	}
	if len(g.Classes) != len(expected) || len(g.Kills) != len(expected) {
		return fail()
	}
	classes := map[string]bool{}
	for _, c := range g.Classes {
		if !expected[c] || classes[c] {
			return fail()
		}
		classes[c] = true
	}
	killed := map[string]DamageEvent{}
	for _, e := range events {
		if e.Spawncount != id.Spawncount || e.Map != id.Map || e.Frame <= release.Frame || e.Frame > id.Frame || e.HealthBefore <= 0 || e.HealthAfter > 0 {
			continue
		}
		if e.Target == id.Actor && e.TargetClass == "player" {
			return fail()
		}
		if e.Attacker == id.Actor && e.AttackerClass == "player" && e.Mod != 21 && expected[e.TargetClass] {
			if _, found := killed[e.TargetClass]; !found {
				killed[e.TargetClass] = e
			}
		}
	}
	if len(killed) != len(expected) {
		return fail()
	}
	last := 0
	targets := map[int]bool{}
	seen := map[string]bool{}
	for _, k := range g.Kills {
		e, found := killed[k.Class]
		if !found || seen[k.Class] || targets[k.Target] || e.Frame != k.Frame || e.Target != k.Target {
			return fail()
		}
		seen[k.Class], targets[k.Target] = true, true
		if k.Frame > last {
			last = k.Frame
		}
	}
	if last != g.KillFrame {
		return fail()
	}
	return nil
}
