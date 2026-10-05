package learningenv

import (
	"testing"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

func capture(frame, life int, health int16) policy.Capture {
	id := policy.Identity{Map: "base1", Connection: 1, Spawncount: 1, Actor: 1, Frame: frame, Life: life}
	cmd := quake.UserCmd{Msec: 100, Forward: 80}
	act := policy.Action{Version: policy.ActionVersion, Identity: id, Forward: .2, Vertical: "release"}
	return policy.Capture{Provider: "rules", Observation: policy.Observation{Version: policy.ObservationVersion, Identity: id, Health: health, PreviousCommand: cmd}, Proposed: act, Applied: act, AppliedCommand: cmd, LabelQuality: "unreviewed"}
}

func TestDeathEndsLifeAndRespawnCannotBootstrap(t *testing.T) {
	a := Assembler{Worker: "worker-3", Episode: "seed-101"}
	if s, _, err := a.Push(capture(5, 1, 100)); err != nil || s != nil {
		t.Fatal(s, err)
	}
	s, o, err := a.Push(capture(6, 1, 0))
	if err != nil || !s.Terminal || s.Truncated || !o.ObservedDeath || *o.HealthDelta != -100 || o.Score != nil {
		t.Fatal(s, o, err)
	}
	if s, _, err = a.Push(capture(7, 2, 100)); err != nil || s != nil {
		t.Fatal("crossed respawn", s, err)
	}
	s, o = a.Close("time_limit")
	if !s.Truncated || s.Terminal || s.Next != nil || o.HealthDelta != nil || s.Observation.Identity.Life != 2 {
		t.Fatal(s, o)
	}
}

func TestInvalidTransitionsAreTruncatedWithoutNext(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*policy.Capture)
	}{
		{"frame_gap", func(c *policy.Capture) { c.Observation.Identity.Frame++; c.Applied.Identity.Frame++ }},
		{"life_or_world_changed", func(c *policy.Capture) { c.Observation.Identity.Connection++; c.Applied.Identity.Connection++ }},
		{"command_history_mismatch", func(c *policy.Capture) { c.Observation.PreviousCommand.Forward = 123 }},
		{"stale_next_observation", func(c *policy.Capture) { c.Observation.AgeMS = 301 }},
		{"harness_override", func(c *policy.Capture) { c.LimitReason = "test_teleport_settling" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := Assembler{}
			_, _, _ = a.Push(capture(1, 1, 100))
			c := capture(2, 1, 90)
			tc.change(&c)
			s, o, err := a.Push(c)
			if err != nil || !s.Truncated || s.Terminal || s.Next != nil || s.Reason != tc.name || o.HealthDelta != nil {
				t.Fatal(s, o, err)
			}
		})
	}
}

func TestInterventionAndHandoffPreservePolicyIntent(t *testing.T) {
	a := Assembler{}
	c := capture(1, 1, 100)
	c.Provider = "probe"
	c.Selection = &policy.Selection{Owner: "provider", Interventions: []policy.Intervention{{Component: "movement", Reason: "wall"}}}
	c.Proposed.Forward = 1
	c.Applied.Forward = 0
	c.AppliedCommand.Forward = 0
	_, _, _ = a.Push(c)
	d := capture(2, 1, 98)
	d.Observation.PreviousCommand = c.AppliedCommand
	s, o, err := a.Push(d)
	if err != nil || !s.Truncated || s.Reason != "control_handoff" || s.Action.Forward != 1 || s.AppliedAction.Forward != 0 || len(s.Interventions) != 1 || *o.HealthDelta != -2 {
		t.Fatal(s, o, err)
	}
}
