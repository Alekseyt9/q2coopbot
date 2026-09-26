package harness

import (
	"q2coopbot/internal/harness/netfault"
	"testing"
)

func TestNetworkFramesRequireExactDroppedEvidence(t *testing.T) {
	seq := uint32(10)
	var events []netfault.Event
	for frame := 10; frame <= 14; frame++ {
		action, stage := "forward", "before"
		if frame == 11 || frame == 12 {
			action, stage = "drop", "blackout"
		} else if frame >= 13 {
			stage = "after"
		}
		events = append(events, netfault.Event{Direction: "server_to_client", Action: action, Stage: stage, Sequence: &seq, Frames: []netfault.GameFrame{{Map: "base1", Generation: 7, Frame: frame}}})
	}
	rows := []Trace{{Map: "base1", Generation: 7, Connection: 1, Frame: 10}, {Map: "base1", Generation: 7, Connection: 1, Frame: 13}, {Map: "base1", Generation: 7, Connection: 1, Frame: 14}}
	if r := VerifyNetworkFrames(events, rows); !r.Accepted || r.ExplainedMissingFrames != 2 || *r.FirstRecoveredFrame != 13 {
		t.Fatal(r)
	}
	for _, test := range []struct {
		name string
		edit func([]netfault.Event, []Trace) ([]netfault.Event, []Trace)
	}{
		{"missing drop", func(e []netfault.Event, r []Trace) ([]netfault.Event, []Trace) { return append(e[:1], e[2:]...), r }},
		{"forwarded but absent", func(e []netfault.Event, r []Trace) ([]netfault.Event, []Trace) { e[1].Action = "forward"; return e, r }},
		{"decode failure", func(e []netfault.Event, r []Trace) ([]netfault.Event, []Trace) {
			e[1].DecodeError = "bad packet"
			return e, r
		}},
		{"dropped yet observed", func(e []netfault.Event, r []Trace) ([]netfault.Event, []Trace) { r[1].Frame = 12; return e, r }},
		{"unexplained gap later", func(e []netfault.Event, r []Trace) ([]netfault.Event, []Trace) { return e, []Trace{r[0], r[2]} }},
		{"connection changed", func(e []netfault.Event, r []Trace) ([]netfault.Event, []Trace) { r[1].Connection = 2; return e, r }},
		{"duplicate frame", func(e []netfault.Event, r []Trace) ([]netfault.Event, []Trace) { r[1].Frame = 10; return e, r }},
		{"foreign generation", func(e []netfault.Event, r []Trace) ([]netfault.Event, []Trace) { r[1].Generation = 8; return e, r }},
	} {
		t.Run(test.name, func(t *testing.T) {
			e, r := test.edit(append([]netfault.Event(nil), events...), append([]Trace(nil), rows...))
			if result := VerifyNetworkFrames(e, r); result.Accepted {
				t.Fatal(result)
			}
		})
	}
	if r := VerifyNetworkFrames(events, rows[1:]); r.Accepted || r.State != "fixture_failed" {
		t.Fatal(r)
	}
}
