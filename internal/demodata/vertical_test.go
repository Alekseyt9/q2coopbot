package demodata

import "testing"

func TestVerticalCandidatesRequireActualPoseChange(t *testing.T) {
	for _, action := range []string{"jump", "crouch", "stand", "jump_release"} {
		t.Run(action, func(t *testing.T) {
			s, e, c := teacherFixture()
			c.TeacherPrimitive, c.LimitReason = "vertical_flat_v1", "teacher_vertical_primitive"
			c.Changed = true
			s.AppliedAction.Forward, s.AppliedAction.Side = 0, 0
			s.Command.Forward, s.Command.Side = 0, 0
			s.Observation.Geometry.UpDistance = 64
			switch action {
			case "jump":
				s.AppliedAction.Vertical = "jump"
				s.Command.Up = 400
				s.Next.OnGround = false
				s.Next.Position[2] = 20
				s.Next.Velocity[2] = 190
			case "crouch":
				s.AppliedAction.Vertical = "crouch"
				s.Command.Up = -400
				s.Next.Ducked = true
			case "stand":
				s.Observation.Ducked = true
			case "jump_release":
				s.Observation.OnGround, s.Next.OnGround = false, false
				s.Observation.PreviousCommand.Up = 400
			}
			c.Applied, c.AppliedCommand = s.AppliedAction, s.Command
			r := SelectVersion(s, e, c, "rules", VerticalSelectionVersion)
			if !r.Heads.Vertical || r.Heads.Movement || r.Heads.Aim || r.Heads.Attack || r.Heads.Weapon || r.Quality == "rejected" {
				t.Fatal(action, r)
			}
			if r := Select(s, e, c, "rules"); r.Quality != "rejected" {
				t.Fatal("v1 copied primitive", r)
			}
			s.Next.OnGround, s.Next.Ducked = true, s.Observation.Ducked
			s.Next.Position[2], s.Next.Velocity[2] = s.Observation.Position[2], 0
			s.Observation.OnGround = true
			if r := SelectVersion(s, e, c, "rules", VerticalSelectionVersion); r.Quality != "rejected" {
				t.Fatal("no actual pose change", r)
			}
		})
	}
}
