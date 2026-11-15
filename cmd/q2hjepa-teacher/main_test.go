package main

import (
	"math"
	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
	"testing"
)

func TestTeacherEnsembleUsesCircularYawAndSoftCategories(t *testing.T) {
	samples := []policy.Action{
		{Forward: 1, YawDelta: 179, Attack: true, Vertical: "release"},
		{Forward: -1, YawDelta: -179, Attack: false, Vertical: "jump"},
	}
	mean := meanPrimitive(samples)
	if math.Abs(math.Abs(mean[2])-1) > 1e-9 {
		t.Fatalf("averaged yaw crossed the wrong direction: %v", mean)
	}
	if mean[0] != 0 || mean[4] != .5 || mean[5] != .5 || mean[6] != .5 || mean[7] != 0 {
		t.Fatalf("soft action mixture wrong: %v", mean)
	}
	if samples[0].YawDelta != 179 || samples[1].YawDelta != -179 {
		t.Fatal("source executed actions changed")
	}
}

func TestNativeReleaseDoesNotAssumeClientFrame100(t *testing.T) {
	id := policy.Identity{Map: "base1", Frame: 98, Actor: 1, Spawncount: 2, Connection: 1, Life: 1}
	next := policy.Observation{Identity: id}
	next.Identity.Frame = 99
	s := learningenv.Step{Observation: policy.Observation{Identity: id}, Next: &next, Owner: "provider",
		Execution: &learningenv.Execution{Matched: true, WindowExclusive: true}, Native: &learningenv.NativeStep{Spawncount: 2, Actor: 1, BeginFrame: 98, EndFrame: 99}}
	if !teacherEligible(s) {
		t.Fatal("first verified native action was incorrectly dropped")
	}
	s.Native = nil
	if teacherEligible(s) {
		t.Fatal("unverified action admitted")
	}
}
