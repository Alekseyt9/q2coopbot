package netfault

import (
	"os"
	"path/filepath"
	"testing"
)

func reportFixture() (Config, []Event) {
	c := Config{Listen: "127.0.0.1:29061", Server: "127.0.0.1:29060", AfterMS: 100, DurationMS: 50}
	seq := uint32(3)
	rows := []Event{{Direction: "client_to_server", Stage: "unarmed", Action: "forward", Bytes: 8}}
	for _, ms := range []int64{0, 100, 149, 150} {
		for _, dir := range []string{"server_to_client", "client_to_server"} {
			stage, action := "before", "forward"
			if ms >= 100 && ms < 150 {
				stage, action = "blackout", "drop"
			} else if ms >= 150 {
				stage = "after"
			}
			rows = append(rows, Event{ElapsedMS: ms, Direction: dir, Stage: stage, Action: action, Bytes: 8, Sequence: &seq})
		}
	}
	return c, rows
}

func TestTransportReportEvidence(t *testing.T) {
	c, rows := reportFixture()
	if r := Analyze(c, rows, 20, true); !r.Accepted || *r.Directions["server_to_client"].RecoveryMS != 0 {
		t.Fatal(r)
	}
	for _, test := range []struct {
		name  string
		edit  func([]Event) []Event
		state string
	}{
		{"wrong action", func(r []Event) []Event { r[3].Action = "forward"; return r }, "trace_invalid"},
		{"wrong stage", func(r []Event) []Event { r[3].Stage = "before"; return r }, "trace_invalid"},
		{"backwards clock", func(r []Event) []Event { r[5].ElapsedMS = 1; return r }, "trace_invalid"},
		{"missing origin", func(r []Event) []Event { return r[3:] }, "trace_invalid"},
		{"not exercised", func(r []Event) []Event { return append(r[:3], r[7:]...) }, "fixture_failed"},
		{"truncated", func(r []Event) []Event { return r[:7] }, "fixture_failed"},
		{"late recovery", func(r []Event) []Event { r[7].ElapsedMS = 171; r[8].ElapsedMS = 171; return r }, "fixture_failed"},
		{"unarmed after arm", func(r []Event) []Event { r[2].Stage = "unarmed"; return r }, "trace_invalid"},
	} {
		t.Run(test.name, func(t *testing.T) {
			r := Analyze(c, test.edit(append([]Event(nil), rows...)), 20, true)
			if r.Accepted || r.State != test.state {
				t.Fatal(r)
			}
		})
	}
	var idle []Event
	for _, r := range rows {
		if r.Direction == "client_to_server" && r.Stage == "blackout" {
			continue
		}
		idle = append(idle, r)
	}
	if r := Analyze(c, idle, 20, false); !r.Accepted || r.Directions["client_to_server"].Dropped != 0 {
		t.Fatal(r)
	}
	if r := Analyze(c, idle, 20, true); r.Accepted {
		t.Fatal("missing upstream loss accepted")
	}
}

func TestReadEventsRejectsIncompleteOrMalformedRows(t *testing.T) {
	for _, data := range []string{`{}`, `{"elapsed_ms":null,"direction":"client_to_server","action":"forward","stage":"unarmed","bytes":4}`, `{"elapsed_ms":0,"direction":"client_to_server","action":"forward","stage":"unarmed","bytes":4,"typo":1}`, `{"elapsed_ms":`, "\n"} {
		path := filepath.Join(t.TempDir(), "network.jsonl")
		if err := os.WriteFile(path, []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := ReadEvents(path); err == nil {
			t.Fatalf("accepted %q", data)
		}
	}
}
