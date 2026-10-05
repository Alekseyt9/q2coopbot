package demodata

import "testing"

func TestReleaseOnlyLabelsAttackAndKeepsV1Frozen(t *testing.T) {
	s, e, c := teacherFixture()
	s.Index = 10
	s.Observation.Enemies, s.Next.Enemies = nil, nil
	if r := Select(s, e, c, "rules"); r.Quality != "rejected" {
		t.Fatal("v1 behavior changed", r)
	}
	r := SelectVersion(s, e, c, "rules", ReleaseSelectionVersion)
	if !r.Heads.Attack || r.Heads.Aim || r.Heads.Movement || r.Heads.Vertical || r.Heads.Weapon || r.Quality == "rejected" {
		t.Fatal(r)
	}
}

func TestReleaseRejectsUncertainOrUnsafeLabels(t *testing.T) {
	for _, name := range []string{"sample", "guard", "proposal", "attack", "dispatch", "damage", "switch", "stale", "setup", "terminal"} {
		t.Run(name, func(t *testing.T) {
			s, e, c := teacherFixture()
			s.Index = 10
			s.Observation.Enemies, s.Next.Enemies = nil, nil
			switch name {
			case "sample":
				s.Index = 11
			case "guard":
				c.MoveLimitReason = "wall"
			case "proposal":
				c.Proposed.Attack = true
			case "attack":
				s.AppliedAction.Attack = true
				c.Applied = s.AppliedAction
			case "dispatch":
				s.Execution.Matched = false
			case "damage":
				e.MonsterHealthDamage = 10
			case "switch":
				s.Next.Weapon = "Shotgun"
			case "stale":
				s.Observation.AgeMS = 400
			case "setup":
				c.LimitReason = "test_combat_barrier"
			case "terminal":
				s.Terminal = true
			}
			if r := SelectVersion(s, e, c, "rules", ReleaseSelectionVersion); r.Quality != "rejected" {
				t.Fatal(name, r)
			}
		})
	}
}
