package learningenv

import (
	"encoding/json"
	"q2coopbot/internal/harness"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"strings"
	"testing"
)

func TestPairedBindingRejectsMissingOrChangedPeer(t *testing.T) {
	pairs, err := ReadPairedNativeSteps(strings.NewReader(pairedFixture()), nil)
	if err != nil {
		t.Fatal(err)
	}
	var traces [2][]harness.Trace
	for role := 0; role < 2; role++ {
		p, c := pairs.Pairs[0].Steps[role], pairs.Pairs[0].Commands[role]
		traces[role] = []harness.Trace{{Connection: 1, Generation: p.Spawncount, Frame: p.BeginFrame, SelfEntity: p.Actor, ClientSequence: p.Sequence, Command: c.Command}}
	}
	native, commands, err := pairs.BindTraces(traces, 1)
	if err != nil || native.Steps[0].Actor != 2 || commands[0].Command.Forward != 10 {
		t.Fatal(native, commands, err)
	}
	original := traces[1][0]
	traces[1] = nil
	if _, _, err = pairs.BindTraces(traces, 0); err == nil {
		t.Fatal("missing peer accepted")
	}
	traces[1] = []harness.Trace{original}
	traces[1][0].Command.Forward++
	if _, _, err = pairs.BindTraces(traces, 0); err == nil {
		t.Fatal("changed peer command accepted")
	}
	traces[1][0] = original
	traces[1][0].SelfEntity = 1
	if _, _, err = pairs.BindTraces(traces, 0); err == nil {
		t.Fatal("duplicate actor accepted")
	}
	traces[1][0] = original
	traces[1][0].Connection = 2
	if _, _, err = pairs.BindTraces(traces, 0); err == nil {
		t.Fatal("reconnection accepted as reset")
	}
}

func TestPairedFinalObservationIsNotAnAppliedCommand(t *testing.T) {
	id := policy.Identity{Connection: 1, Spawncount: 42, Frame: 100, Actor: 2, Life: 1, Map: "base1"}
	action := policy.Action{Version: policy.ActionVersion, Identity: id}
	capture := policy.Capture{Observation: policy.Observation{Version: policy.ObservationVersion, Identity: id}, Applied: action, Proposed: action, Provider: "terminal_observer"}
	row := map[string]any{"terminal_observation_only": true, "connection": 1, "spawncount": 42, "frame": 100, "observation_frame": 100, "self_entity": 2, "map": "base1", "health": 0, "self": quake.Vec3{}, "combat_policy": capture}
	data, _ := json.Marshal(row)
	if rows, err := ReadPairedTrace(strings.NewReader(string(data))); err != nil || len(rows) != 1 || !rows[0].TerminalObservationOnly {
		t.Fatal(rows, err)
	}
	if _, err := ReadPairedTrace(strings.NewReader(string(data) + "\n" + string(data))); err == nil {
		t.Fatal("trace continued after terminal observation")
	}
	row["client_sequence"] = 1
	data, _ = json.Marshal(row)
	if _, err := ReadPairedTrace(strings.NewReader(string(data))); err == nil {
		t.Fatal("terminal accepted as sent command")
	}
}

func TestPairedTraceEnvelopeRequiresExactObservation(t *testing.T) {
	id := policy.Identity{Connection: 1, Spawncount: 42, Frame: 98, Actor: 1, Life: 1, Map: "base1"}
	capture := policy.Capture{Observation: policy.Observation{Version: policy.ObservationVersion, Identity: id}, Applied: policy.Action{Version: policy.ActionVersion, Identity: id}}
	row := map[string]any{"connection": 1, "spawncount": 42, "frame": 98, "observation_frame": 98, "self_entity": 1, "map": "base1", "client_sequence": 19, "combat_policy": capture, "sent_command": capture.AppliedCommand}
	data, _ := json.Marshal(row)
	if _, err := ReadPairedTrace(strings.NewReader(string(data))); err != nil {
		t.Fatal(err)
	}
	row["observation_frame"] = 99
	data, _ = json.Marshal(row)
	if _, err := ReadPairedTrace(strings.NewReader(string(data))); err == nil {
		t.Fatal("mismatched observation accepted")
	}
	row["observation_frame"] = 98
	capture.Applied.Identity.Actor = 2
	row["combat_policy"] = capture
	data, _ = json.Marshal(row)
	if _, err := ReadPairedTrace(strings.NewReader(string(data))); err == nil {
		t.Fatal("mismatched applied identity accepted")
	}
}
