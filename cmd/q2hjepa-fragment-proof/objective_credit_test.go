package main

import (
	"fmt"
	"strings"
	"testing"

	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
)

func objectiveCreditFixture() (string, string, policy.Identity, []learningenv.FragmentTransition) {
	root := policy.Identity{Map: "base1", Spawncount: 7, Frame: 110}
	rows := make([]learningenv.FragmentTransition, 3)
	tail := "g_test_objective ready version=1\n"
	for i := range rows {
		id := policy.Identity{Map: "base1", Spawncount: 9, Actor: 1, Life: 1, Frame: i + 1}
		next := id
		next.Frame++
		rows[i] = learningenv.FragmentTransition{Identity: id, NextIdentity: next, NextHealth: 100}
		alive, pending := 0, 0
		if i == 0 {
			pending = 1 // Dead first boss, next phase not yet spawned.
		}
		if i == 1 {
			alive = 1 // A revived/new phase actor still lives.
		}
		tail += fmt.Sprintf("sv_test_objective spawncount=9 server_frame=%d g_test_objective version=1 map=base1 game_frame=%d alive=%d pending=%d\n", i+2, i+113, alive, pending)
	}
	source := "g_test_objective ready version=1\nsv_test_objective spawncount=7 server_frame=110 g_test_objective version=1 map=base1 game_frame=112 alive=1 pending=0\n"
	return source, tail, root, rows
}

func TestFragmentNativeObjectiveWaitsForPendingAndRevivedActors(t *testing.T) {
	source, tail, root, rows := objectiveCreditFixture()
	states, err := fragmentObjectiveStates(source, tail, root, rows)
	if err != nil {
		t.Fatal(err)
	}
	// Initial roster cleared, but the native objective disproves completion.
	rows[0].Events = []learningenv.DamageEvent{{Attacker: 1, Target: 7, TargetClass: "monster_medic", HealthBefore: 20, Take: 20, HealthAfter: 0}}
	rewards, step, reason, err := nativeTerminalRewardsWithObjectives(rows, map[int]bool{7: true}, 20, true, states)
	if err != nil || step != 2 || reason != "native_objective_clear" || rewards[0] != 10 || rewards[1] != 0 || rewards[2] != 100 {
		t.Fatalf("premature/missing objective terminal: %v %d %s %v", rewards, step, reason, err)
	}
	if _, _, _, err := nativeTerminalRewardsWithObjectives(rows[:2], map[int]bool{7: true}, 20, true, states); err == nil {
		t.Fatal("pending/living objective replaced by initial-roster closure")
	}
	rows[2].NextHealth = -1
	rows[2].Events = []learningenv.DamageEvent{{Attacker: 7, Target: 1, HealthBefore: 100, Take: 101, HealthAfter: -1}}
	_, _, reason, err = nativeTerminalRewardsWithObjectives(rows, map[int]bool{7: true}, 20, false, states)
	if err != nil || reason != "native_death" {
		t.Fatal("simultaneous objective overrode actor death", reason, err)
	}
}

func TestFragmentObjectiveRejectsMissingWrongAndFutureReceipts(t *testing.T) {
	source, tail, root, rows := objectiveCreditFixture()
	for name, bad := range map[string]string{
		"missing ready":      strings.Replace(tail, "g_test_objective ready version=1\n", "", 1),
		"wrong generation":   strings.Replace(tail, "spawncount=9", "spawncount=8", 1),
		"wrong map":          strings.Replace(tail, "map=base1", "map=base2", 1),
		"future game clock":  strings.Replace(tail, "game_frame=113", "game_frame=114", 1),
		"missing transition": strings.Join(strings.Split(tail, "\n")[:3], "\n"),
		"disabled tail":      "",
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := fragmentObjectiveStates(source, bad, root, rows); err == nil {
				t.Fatal("invalid objective admitted")
			}
		})
	}
	for _, badSource := range []string{"", strings.Replace(source, "server_frame=110", "server_frame=109", 1), strings.Replace(source, "alive=1", "alive=0", 1)} {
		if _, err := fragmentObjectiveStates(badSource, tail, root, rows); err == nil {
			t.Fatal("missing or already complete source objective admitted")
		}
	}
	if states, err := fragmentObjectiveStates("legacy log", "legacy log", root, rows); err != nil || states != nil {
		t.Fatal("legacy captures must retain their distinct conservative proof", err)
	}
}
