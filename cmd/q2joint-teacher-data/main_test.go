package main

import (
	"q2coopbot/internal/policy"
	"testing"
)

func TestTeacherFireReleaseDoesNotCopyOtherCorrections(t *testing.T) {
	c := policy.Capture{Selection: &policy.Selection{}, Changed: true, LimitReason: "barrel_blast_risk"}
	c.Proposed.Attack = true
	c.ProposedCommand.Buttons = 1
	c.AppliedCommand.Msec, c.AppliedCommand.Light = 100, 115
	if !fireCandidate(c) {
		t.Fatal("pure recorded fire release rejected")
	}
	c.AppliedCommand.Pitch = 100
	if fireCandidate(c) {
		t.Fatal("modified aim accepted as pure fire release")
	}
	c.AppliedCommand.Pitch = 0
	c.LimitReason = "unknown_guard"
	if fireCandidate(c) {
		t.Fatal("unknown correction accepted")
	}
}

func TestTeacherIntentUsesOnlyAdjacentSameLife(t *testing.T) {
	p := &policy.TargetIntent{Identity: policy.Identity{Map: "base3", Connection: 1, Spawncount: 3, Actor: 0, Life: 1, Frame: 100}, Entity: 12, Track: 1}
	o := policy.Observation{Identity: p.Identity}
	o.Identity.Frame++
	if pastIntent(p, o) == nil {
		t.Fatal("adjacent history lost")
	}
	o.Identity.Frame++
	if pastIntent(p, o) != nil {
		t.Fatal("gap not reset")
	}
	o.Identity.Frame = 101
	o.Identity.Life++
	if pastIntent(p, o) != nil {
		t.Fatal("life not reset")
	}
}
