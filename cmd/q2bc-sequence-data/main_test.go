package main

import (
	"q2coopbot/internal/aimquery"
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

func TestCounterfactualAimUsesSeparateTargetAndCannotInventFireLabel(t *testing.T) {
	id := policy.Identity{Life: 1, Frame: 101, Map: "base1", Actor: 1}
	o := policy.Observation{Version: policy.ObservationVersion, Identity: id, Health: 100, Weapon: "Blaster"}
	d := diagnostic{Source: demodata.Source{Seed: 42}, Selection: demodata.Selection{Version: demodata.AimQuerySelectionVersion, Quality: "unexecuted query", Heads: demodata.Heads{Aim: true}}, Step: learningenv.Step{Owner: "provider", Observation: o, AppliedAction: policy.Action{YawDelta: 20, Attack: true}, Native: &learningenv.NativeStep{}, Execution: &learningenv.Execution{Matched: true}}, Query: &aimquery.Label{Version: aimquery.Version, Action: policy.Action{Version: policy.ActionVersion, Identity: id, YawDelta: -90, Vertical: "release"}}}
	r, e := convert(d, policy.WeaponFeatureVersion)
	if e != nil || r == nil || r.Targets[2] != -.5 || r.Mask[2] || r.Attack || r.LabelKind != "counterfactual_nominal_aim" {
		t.Fatal(r, e)
	}
	if d.Step.AppliedAction.YawDelta != 20 || !d.Step.AppliedAction.Attack {
		t.Fatal("source action rewritten")
	}
	d.Query = nil
	if _, e = convert(d, policy.WeaponFeatureVersion); e == nil {
		t.Fatal("query absent but label accepted")
	}
}
