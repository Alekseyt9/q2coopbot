package demodata

import "testing"

func TestTrackingRequiresObservableTeacherTargetAndDoesNotChangeV2(t *testing.T) {
	s, e, c := teacherFixture()
	s.Next.Position = s.Observation.Position
	c.TeacherAimSource, c.TeacherAimEntity = "enemy", 2
	clear := true
	s.Observation.Enemies[0].ClearShot = &clear
	r := SelectVersion(s, e, c, "rules", TrackingSelectionVersion)
	if !r.Heads.Aim || !r.Heads.Attack || !r.Heads.Vertical || r.Heads.Movement || r.Quality == "rejected" {
		t.Fatal(r)
	}
	if SelectVersion(s, e, c, "rules", ReleaseSelectionVersion).Quality != "rejected" {
		t.Fatal("changed frozen v2")
	}
	// A delayed hit neither supplies nor rescues the teacher target identity.
	e.MonsterHealthDamage = 10
	for _, kind := range []string{"target", "clear", "source", "guard", "provider"} {
		t.Run(kind, func(t *testing.T) {
			ss, cc := s, c
			switch kind {
			case "target":
				cc.TeacherAimEntity = 77
			case "clear":
				ss.Observation.Enemies = nil
			case "source":
				cc.TeacherAimSource = "route"
			case "guard":
				cc.LimitReason = "friendly_line_of_fire"
			case "provider":
				ss.Owner = "provider"
			}
			if r := SelectVersion(ss, e, cc, "rules", TrackingSelectionVersion); r.Heads.Aim {
				t.Fatal(r)
			}
		})
	}
}
