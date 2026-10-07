package main

import (
	"q2coopbot/internal/demodata"
	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
	"testing"
)

func TestRejectedLabelsRetainVerifiedContext(t *testing.T) {
	d := diagnostic{Source: demodata.Source{Seed: 42}, Selection: demodata.Selection{Quality: "rejected", Heads: demodata.Heads{Aim: true}}, Step: learningenv.Step{Observation: policy.Observation{Version: policy.ObservationVersion, Identity: policy.Identity{Life: 1, Frame: 100, Map: "base1", Connection: 1, Spawncount: 42, Actor: 1}, Health: 100, Weapon: "Blaster"}, AppliedAction: policy.Action{YawDelta: 18}, Native: &learningenv.NativeStep{}, Execution: &learningenv.Execution{Matched: true}}}
	row, err := convert(d, policy.WeaponFeatureVersion)
	if err != nil || row == nil {
		t.Fatalf("lost context: %v", err)
	}
	if len(row.Features) != 845 || row.Mask[1] {
		t.Fatal("wrong contract or unverified aim label", row.Mask)
	}
	d.Selection.Quality = "teacher_tracking"
	row, err = convert(d, policy.WeaponFeatureVersion)
	if err != nil || !row.Mask[1] || row.Targets[2] != .1 {
		t.Fatal("verified aim label lost", err)
	}
	d.Step.Execution.Matched = false
	row, err = convert(d, policy.WeaponFeatureVersion)
	if err != nil || row != nil {
		t.Fatal("unproven context accepted")
	}
}
