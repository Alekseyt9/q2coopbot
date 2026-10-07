package demodata

import (
	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"testing"
)

func TestAimQueryDoesNotRelabelActualActionOrEffects(t *testing.T) {
	clear := true
	solid := uint16(2 | (3 << 5) | (8 << 10))
	o := policy.Observation{Version: policy.ObservationVersion, Identity: policy.Identity{Frame: 101, Life: 1, Map: "base1", Actor: 1}, Health: 60, Weapon: "Blaster", ViewAngles: [3]int16{0, 16384, 0}, Enemies: []policy.Enemy{{ID: 2, Class: "monster_parasite", Relative: quake.Vec3{100, 0, 0}, ClearShot: &clear, Solid: &solid}}}
	s := learningenv.Step{Owner: "provider", Provider: "ppo:test", Observation: o, AppliedAction: policy.Action{YawDelta: 15, Attack: true}, Native: &learningenv.NativeStep{}, Execution: &learningenv.Execution{Matched: true}}
	c := policy.Capture{Provider: s.Provider, Selection: &policy.Selection{Mode: "learned", Owner: "provider"}}
	r, q, err := SelectAimQuery(s, c, "learned")
	if err != nil || q == nil || !r.Heads.Aim || r.Heads.Attack || r.Heads.Movement || r.Heads.Vertical || q.Action.YawDelta != -90 {
		t.Fatal(r, q, err)
	}
	if s.AppliedAction.YawDelta != 15 || !s.AppliedAction.Attack {
		t.Fatal("actual action changed")
	}
	s.Execution.Matched = false
	r, q, err = SelectAimQuery(s, c, "learned")
	if err != nil || q != nil || r.Quality != "rejected" {
		t.Fatal("unproven state labeled", r, q, err)
	}
}
