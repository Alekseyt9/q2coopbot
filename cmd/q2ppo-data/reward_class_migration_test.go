package main

import (
	"q2coopbot/internal/learningenv"
	"testing"
)

func TestClassMigrationOnlyAllowsCommitmentStateDelta(t *testing.T) {
	x, y := .12, .10
	a := learningenv.Reward{Version: learningenv.BlockedGroundRequestRewardVersion, Available: true,
		Step: 1, Score: &x, Components: map[string]float64{"monster_damage": .1, "target_commitment": .02}}
	b := a
	b.Score, b.Components = &y, map[string]float64{"monster_damage": .1, "target_commitment": 0}
	if !commitmentStateDeltaOnly(a, b) {
		t.Fatal("legitimate commitment state delta rejected")
	}
	b.Components["monster_damage"] = .09
	if commitmentStateDeltaOnly(a, b) {
		t.Fatal("base reward change accepted")
	}
	b.Components["monster_damage"] = .1
	b.Step = 2
	if commitmentStateDeltaOnly(a, b) {
		t.Fatal("identity change accepted")
	}
	b.Step = 1
	y = .09
	if commitmentStateDeltaOnly(a, b) {
		t.Fatal("score unrelated to commitment accepted")
	}
}
