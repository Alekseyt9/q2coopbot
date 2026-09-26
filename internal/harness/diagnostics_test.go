package harness

import (
	"q2coopbot/internal/quake"
	"testing"
)

func stalledRows() (Scenario, []Trace) {
	s := fixture()
	s.Steps = []Step{{ID: "walk", Action: "walk"}}
	ground := true
	rows := []Trace{}
	for f := 40; f <= 44; f++ {
		rows = append(rows, Trace{Map: "base2", Frame: f, Generation: 1, Self: &quake.Vec3{}, OnGround: &ground, Teammate: &quake.Vec3{32, 0, 0}, Command: quake.UserCmd{Forward: 100}, Scenario: &Status{StepID: "walk", State: "running"}})
	}
	return s, rows
}
func TestStallEvidenceUsesConsecutiveGroundedCommands(t *testing.T) {
	s, rows := stalledRows()
	d := diagnoseMovement(s, rows)
	if len(d.Stalls) != 1 {
		t.Fatal(d)
	}
	v := d.Stalls[0]
	if v.Start != 40 || v.Detected != 43 || v.End != 44 || v.Intervals != 4 || len(v.Nearby) != 1 || v.Nearby[0].Frames != 4 {
		t.Fatal(v)
	}
	for _, tc := range []struct {
		name   string
		mutate func([]Trace)
	}{
		{"no_command", func(r []Trace) {
			for i := range r {
				r[i].Command = quake.UserCmd{}
			}
		}},
		{"airborne", func(r []Trace) {
			off := false
			for i := range r {
				r[i].OnGround = &off
			}
		}},
		{"missing_ground", func(r []Trace) {
			for i := range r {
				r[i].OnGround = nil
			}
		}},
		{"actual_progress", func(r []Trace) {
			for i := range r {
				r[i].Self = &quake.Vec3{float64(i) * 2, 0, 0}
			}
		}},
		{"gap", func(r []Trace) { r[2].Frame = 100 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, r := stalledRows()
			tc.mutate(r)
			if len(diagnoseMovement(s, r).Stalls) != 0 {
				t.Fatal("false stall")
			}
		})
	}
}
func TestNearbyObservationIsOptionalAndNotInferredFromMemory(t *testing.T) {
	s, r := stalledRows()
	for i := range r {
		r[i].Teammate = nil
		r[i].Enemies = []ObservedEnemy{{ID: 7, Origin: quake.Vec3{40, 0, 0}}, {ID: 8, Origin: quake.Vec3{10, 0, 100}}}
	}
	d := diagnoseMovement(s, r)
	if len(d.Stalls) != 1 || len(d.Stalls[0].Nearby) != 1 || d.Stalls[0].Nearby[0].Entity != 7 {
		t.Fatal(d)
	}
	for i := range r {
		r[i].Enemies = nil
	}
	if len(diagnoseMovement(s, r).Stalls[0].Nearby) != 0 {
		t.Fatal("invented obstacle")
	}
}

func TestExpectedStallRequiresEvidenceInFailedStep(t *testing.T) {
	s, r := stalledRows()
	d := diagnoseMovement(s, r)
	f := &ExpectedFailure{StepID: "walk", MinStallFrames: 3, NearbyKind: "teammate"}
	if !matchesStallEvidence(f, d) {
		t.Fatal("valid evidence rejected")
	}
	if matchesStallEvidence(f, nil) {
		t.Fatal("missing evidence accepted")
	}
	f.StepID = "other"
	if matchesStallEvidence(f, d) {
		t.Fatal("evidence from other step accepted")
	}
	f.StepID = "walk"
	f.NearbyKind = "enemy"
	if matchesStallEvidence(f, d) {
		t.Fatal("wrong nearby kind accepted")
	}
	f.NearbyKind = "teammate"
	f.MinStallFrames = 5
	if matchesStallEvidence(f, d) {
		t.Fatal("short evidence accepted")
	}
}
