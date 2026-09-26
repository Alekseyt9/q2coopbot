package harness

import "testing"

func TestTimelineReloadKeepsFrameIdentities(t *testing.T) {
	rows := []Trace{{Map: "base2", Generation: 2, Frame: 40}, {Map: "base2", Generation: 2, Frame: 41}, {Map: "base2", Generation: 3, Frame: 1}, {Map: "base2", Generation: 3, Frame: 2}}
	timeline := traceTimeline(rows)
	if len(timeline.Segments) != 2 || len(timeline.Issues) != 0 || timeline.Segments[1].First.Row != 3 || timeline.Segments[1].First.Generation != 3 || timeline.Segments[0].Rows != 2 {
		t.Fatalf("%+v", timeline)
	}
}

func TestTimelineDiscontinuities(t *testing.T) {
	for _, tc := range []struct {
		name string
		next Trace
		kind string
	}{
		{"duplicate", Trace{Map: "base2", Generation: 2, Frame: 40}, "duplicate_frame"},
		{"reverse", Trace{Map: "base2", Generation: 2, Frame: 1}, "frame_reversed"},
		{"gap", Trace{Map: "base2", Generation: 2, Frame: 42}, "frame_gap"},
		{"map", Trace{Map: "base1", Generation: 2, Frame: 41}, "map_changed_without_generation"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := traceTimeline([]Trace{{Map: "base2", Generation: 2, Frame: 40}, tc.next})
			if len(got.Issues) != 1 || got.Issues[0].Kind != tc.kind || got.Issues[0].At.Row != 2 || got.Issues[0].Previous.Frame != 40 {
				t.Fatalf("%+v", got)
			}
		})
	}
	got := traceTimeline([]Trace{{Map: "base2", Generation: 2}, {Map: "base1", Generation: 3}, {Map: "base2", Generation: 2}})
	if len(got.Issues) != 1 || got.Issues[0].Kind != "generation_revisited" {
		t.Fatalf("%+v", got)
	}
}

func TestMapResetFailureIsNotScenarioIncomplete(t *testing.T) {
	s := fixture()
	actor := []Trace{{Map: "base2", Generation: 2, Frame: 40}, {Map: "base1", Generation: 3, Frame: 1, Scenario: &Status{State: "failed", StepID: "wait", Reason: "map_generation_changed", EndFrame: 1}}}
	r := Analyze(s, actor, nil)
	if r.Accepted || r.State != "trace_invalid" || r.Reason != "map_generation_changed" || r.ProblemLocation == nil || r.ProblemLocation.Generation != 3 || r.ProblemLocation.Row != 2 {
		t.Fatalf("%+v", r)
	}
}

func TestScenarioWindowCannotSkipForeignMap(t *testing.T) {
	s := fixture()
	rows := []Trace{{Map: "base2", Generation: 2, Frame: 40}, {Map: "base1", Generation: 3, Frame: 1}, {Map: "base2", Generation: 2, Frame: 41}, {Map: "base2", Generation: 2, Frame: 42, Scenario: &Status{State: "completed", EndFrame: 42, CompletedSteps: 1}}}
	r := Analyze(s, rows, rows)
	if r.Accepted || r.State != "trace_invalid" {
		t.Fatalf("%+v", r)
	}
}
