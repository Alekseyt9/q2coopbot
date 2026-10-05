package demodata

import (
	"testing"

	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

func teacherFixture() (learningenv.Step, learningenv.ServerOutcome, policy.Capture) {
	id := policy.Identity{Life: 1, Map: "base1", Connection: 1, Spawncount: 42, Actor: 1, Frame: 10}
	o := policy.Observation{Version: policy.ObservationVersion, Identity: id, Health: 100, Weapon: "Blaster", OnGround: true, Geometry: &policy.LocalGeometry{}, Enemies: []policy.Enemy{{ID: 2, Class: "monster_parasite"}}}
	next := o
	next.Identity.Frame++
	next.Position = quake.Vec3{3, 0, 0}
	a := policy.Action{Version: policy.ActionVersion, Identity: id, Forward: .5, Vertical: "release"}
	cmd := quake.UserCmd{Forward: 200, Msec: 100}
	s := learningenv.Step{Version: learningenv.StepVersion, Worker: "w", Episode: "e", Index: 1, ClientSequence: 23, Observation: o, Next: &next, Action: a, AppliedAction: a, Command: cmd, Owner: "rules", Provider: "rules", Execution: &learningenv.Execution{Matched: true, WindowExclusive: true}, Native: &learningenv.NativeStep{Spawncount: 42, Actor: 1, Sequence: 23, BeginFrame: 10, EndFrame: 11}}
	e := learningenv.ServerOutcome{Version: "server_step_effects_v1", Worker: "w", Episode: "e", Step: 1, Available: true}
	c := policy.Capture{Provider: "rules", Selection: &policy.Selection{Mode: "rules", Owner: "rules"}, Observation: o, Applied: a, AppliedCommand: cmd}
	return s, e, c
}

func TestSelectUsesActualMovementAndComponentMasks(t *testing.T) {
	s, e, c := teacherFixture()
	r := Select(s, e, c, "rules")
	if !r.Heads.Movement || r.Heads.Aim || r.Heads.Attack || r.Heads.Vertical || r.Heads.Weapon || r.Quality == "rejected" {
		t.Fatal(r)
	}
	s.Next.Position = s.Observation.Position
	r = Select(s, e, c, "rules")
	if r.Quality != "rejected" {
		t.Fatal("copied blocked movement", r)
	}
}
func TestSelectDoesNotAttributeDelayedProjectileDamageToAim(t *testing.T) {
	s, e, c := teacherFixture()
	s.Next.Position = s.Observation.Position
	s.AppliedAction.Attack = true
	c.Applied = s.AppliedAction
	e.MonsterHealthDamage = 10
	e.Events = []learningenv.DamageEvent{{Mod: 1, Attacker: 1, Inflictor: 77, Target: 2, TargetClass: "monster_parasite", HealthBefore: 100, Take: 10}}
	if r := Select(s, e, c, "rules"); r.Quality != "rejected" {
		t.Fatal(r)
	}
	s.Observation.Weapon = "Shotgun"
	s.Next.Weapon = "Shotgun"
	e.Events[0].Mod = 2
	e.Events[0].Inflictor = 1
	r := Select(s, e, c, "rules")
	if !r.Heads.Aim || !r.Heads.Attack || r.Heads.Movement {
		t.Fatal(r)
	}
	e.Events[0].HealthBefore = 0
	if r := Select(s, e, c, "rules"); r.Quality != "rejected" {
		t.Fatal("copied corpse damage", r)
	}
}
func TestSelectionRejectsUnsafeAndDiagnosticLabels(t *testing.T) {
	for _, name := range []string{"probe", "shadow", "respawn", "death", "truncated", "gap", "guard", "hurt", "friendly", "self", "dispatch", "bad_action"} {
		t.Run(name, func(t *testing.T) {
			s, e, c := teacherFixture()
			mode := "rules"
			switch name {
			case "probe":
				s.Owner = "provider"
				s.Provider = "diagnostic_probe"
			case "shadow":
				mode = "learned-shadow"
			case "respawn":
				s.Observation.Identity.Life = 2
			case "death":
				s.Terminal = true
				s.Next.Health = 0
			case "truncated":
				s.Truncated = true
			case "gap":
				s.Next.Identity.Frame++
			case "guard":
				c.MoveLimitReason = "wall"
			case "hurt":
				e.ReceivedHealthDamage = 1
			case "friendly":
				e.TeammateHealthDamage = 1
			case "self":
				e.SelfHealthDamage = 1
			case "dispatch":
				s.Execution.WindowExclusive = false
			case "bad_action":
				s.AppliedAction.Forward = 2
				c.Applied = s.AppliedAction
			}
			if r := Select(s, e, c, mode); r.Quality != "rejected" {
				t.Fatal(name, r)
			}
		})
	}
}
func TestSpecPreventsSeedLeakage(t *testing.T) {
	s := Spec{Version: "combat_dataset_spec_v1", SelectionVersion: SelectionVersion, Condition: Condition{Map: "base1", Synchronous: true}, Episodes: []EpisodeSpec{{1, "train"}, {2, "validation"}, {3, "test"}}}
	if err := s.Validate(); err != nil {
		t.Fatal(err)
	}
	s.Episodes[2].Seed = 1
	if s.Validate() == nil {
		t.Fatal("same seed accepted across train/test")
	}
	s.Episodes[2].Seed = 3
	s.Episodes[2].Split = "train"
	if s.Validate() == nil {
		t.Fatal("missing test split accepted")
	}
}
