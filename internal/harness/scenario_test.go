package harness

import (
	"q2coopbot/internal/quake"
	"testing"
)

func fixture() Scenario {
	return Scenario{Version: 1, Name: "test", Map: "base2", StartFrame: 40, GameFrames: 100, Steps: []Step{{ID: "wait", Action: "wait", Frames: 2}}}
}
func input(frame int) Input {
	return Input{Frame: frame, Generation: 2, Map: "base2", Health: 100, OnGround: true}
}

func TestPlaceRequiresObservedArrivalAndRunsOnce(t *testing.T) {
	s := fixture()
	target := quake.Vec3{100, 0, 0}
	s.Steps = []Step{{ID: "place", Action: "place", Target: &target, Timeout: 3}}
	r := New(s)
	if r.Tick(input(40)).Place == nil {
		t.Fatal("missing placement")
	}
	if r.Tick(input(40)).Place != nil {
		t.Fatal("duplicate placement")
	}
	r.Tick(input(41))
	if r.Status.State != "running" {
		t.Fatal(r.Status)
	}
	in := input(42)
	in.Self = target
	r.Tick(in)
	if r.Status.State != "completed" || r.Status.EndFrame != 42 {
		t.Fatal(r.Status)
	}
}
func TestRunnerFailures(t *testing.T) {
	for _, tc := range []struct {
		name   string
		in     Input
		reason string
	}{
		{"gap", input(42), "frame_discontinuity"},
		{"generation", Input{Frame: 41, Generation: 3, Map: "base2", Health: 100}, "map_generation_changed"},
		{"death", Input{Frame: 41, Generation: 2, Map: "base2"}, "actor_dead"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := New(fixture())
			r.Tick(input(40))
			r.Tick(tc.in)
			if r.Status.Reason != tc.reason {
				t.Fatal(r.Status)
			}
			r.Tick(input(43))
			if r.Status.State != "failed" {
				t.Fatal(r.Status)
			}
		})
	}
}
func TestWalkTimeoutAndGround(t *testing.T) {
	s := fixture()
	target := quake.Vec3{100, 0, 0}
	s.Steps = []Step{{ID: "walk", Action: "walk", Target: &target, Timeout: 2}}
	r := New(s)
	r.Tick(input(40))
	in := input(41)
	in.Self = target
	in.OnGround = false
	r.Tick(in)
	if r.Status.State != "running" {
		t.Fatal(r.Status)
	}
	r.Tick(input(42))
	if r.Status.Reason != "step_timeout" {
		t.Fatal(r.Status)
	}
}
func TestValidation(t *testing.T) {
	s := fixture()
	if err := s.Validate(); err != nil {
		t.Fatal(err)
	}
	s.StartFrame = 10001
	if s.Validate() == nil {
		t.Fatal("accepted late start")
	}
	s = fixture()
	s.Steps = append(s.Steps, s.Steps[0])
	if s.Validate() == nil {
		t.Fatal("duplicate ID")
	}
}

func TestRouteRejectionIsTerminal(t *testing.T) {
	s := fixture()
	target := quake.Vec3{100, 0, 0}
	s.Steps = []Step{{ID: "walk", Action: "walk", Target: &target, Route: true, Timeout: 10}}
	r := New(s)
	r.Tick(input(40))
	r.RejectRoute("not_grounded")
	if r.Status.State != "running" {
		t.Fatal("temporary movement condition terminated scenario")
	}
	r.RejectRoute("route_target_outside_aas")
	if r.Status.State != "failed" || r.Status.EndFrame != 40 || r.Status.StepID != "walk" {
		t.Fatal(r.Status)
	}
	if d := r.Tick(input(41)); d.Walk != nil || d.Place != nil {
		t.Fatal("action after rejected route")
	}
	r.RejectRoute("route_unavailable")
	if r.Status.Reason != "route_target_outside_aas" {
		t.Fatal("first failure overwritten")
	}
}
func TestReportMetricsAndIntegrity(t *testing.T) {
	s := fixture()
	s.Expect = Expectations{ContactLosses: 1, Reacquisitions: 1, FollowResumptions: 1}
	actor := []Trace{}
	bot := []Trace{}
	pos := quake.Vec3{}
	for f := 40; f <= 44; f++ {
		actor = append(actor, Trace{Map: s.Map, Generation: 2, Frame: f})
		bot = append(bot, Trace{Map: s.Map, Generation: 2, Frame: f, Goal: "search"})
	}
	actor[4].Scenario = &Status{State: "completed", EndFrame: 44, CompletedSteps: 1}
	bot[0].Teammate = &pos
	bot[3].Teammate = &pos
	bot[4].Teammate = &pos
	bot[4].Goal = "follow_teammate"
	bot[4].Command.Forward = 100
	r := Analyze(s, actor, bot)
	if r.State != "passed" || r.Metrics.RecoveryFrames[0] != 2 || r.Metrics.FollowDelayFrames[0] != 1 || r.Metrics.VisibleFrames != 3 {
		t.Fatalf("%+v", r)
	}
	for i := range bot {
		bot[i].Generation = 3
	}
	if Analyze(s, actor, bot).State != "trace_invalid" {
		t.Fatal("cross generation accepted")
	}
	bot = bot[:4]
	if Analyze(s, actor, bot).State != "trace_invalid" {
		t.Fatal("missing frame accepted")
	}
	actor[4].Scenario.State = "failed"
	bot = append(bot, Trace{Map: s.Map, Frame: 44, Generation: 2})
	for i := range bot {
		bot[i].Generation = 2
	}
	if Analyze(s, actor, bot).State != "fixture_failed" {
		t.Fatal("fixture failure misclassified")
	}
}
