package main

import (
	"strings"
	"testing"

	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
)

func TestNativeTerminalRewardsRequireClosureProof(t *testing.T) {
	rows := []learningenv.FragmentTransition{{Identity: policy.Identity{Actor: 1}, NextHealth: 100},
		{Identity: policy.Identity{Actor: 1}, NextHealth: 100, Events: []learningenv.DamageEvent{
			{Attacker: 1, Target: 7, TargetClass: "monster_soldier", HealthBefore: 20, HealthAfter: -5, Take: 25},
		}}}
	alive := map[int]bool{7: true}
	rewards, terminal, reason, err := nativeTerminalRewards(rows, alive, 20, true)
	if err != nil || terminal != 1 || reason != "native_source_roster_clear" || rewards[0] != 0 || rewards[1] != 110 {
		t.Fatalf("delayed native kill: %v %d %s %v", rewards, terminal, reason, err)
	}
	if !alive[7] {
		t.Fatal("credit calculation mutated caller's checkpoint roster")
	}
	// A medic can revive an initial monster: initial IDs all dead is not a
	// proven final goal for a roster with an unsupported lifecycle.
	if _, _, _, err := nativeTerminalRewards(rows, alive, 20, false); err == nil || !strings.Contains(err.Error(), "bootstrap still required") {
		t.Fatal("unsafe roster clear invented a zero-bootstrap goal", err)
	}
	if _, _, _, err := nativeTerminalRewards(rows[:1], alive, 20, true); err == nil {
		t.Fatal("quiet horizon invented a terminal")
	}
}

func TestNativeDeathWorksWithMedicAndOverridesSimultaneousGoal(t *testing.T) {
	row := learningenv.FragmentTransition{Identity: policy.Identity{Actor: 1}, NextHealth: -3,
		Events: []learningenv.DamageEvent{
			{Attacker: 1, Target: 7, TargetClass: "monster_medic", HealthBefore: 20, HealthAfter: 0, Take: 20},
			{Attacker: 1, Target: 1, HealthBefore: 7, HealthAfter: -3, Take: 10},
		}}
	for _, safe := range []bool{false, true} {
		rewards, terminal, reason, err := nativeTerminalRewards([]learningenv.FragmentTransition{row}, map[int]bool{7: true}, 20, safe)
		// 10 capped monster progress - .7 received - 7 self - 100 death.
		if err != nil || terminal != 0 || reason != "native_death" || rewards[0] != -97.7 {
			t.Fatalf("native death: %v %d %s %v", rewards, terminal, reason, err)
		}
	}
	row.NextHealth = 1
	if _, _, _, err := nativeTerminalRewards([]learningenv.FragmentTransition{row}, map[int]bool{7: true}, 20, false); err == nil {
		t.Fatal("death accepted without next dead observation")
	}
}
