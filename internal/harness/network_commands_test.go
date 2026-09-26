package harness

import (
	"q2coopbot/internal/harness/netfault"
	"q2coopbot/internal/quake"
	"testing"
)

func TestNetworkCommandProofOnlyExcusesProvenDrops(t *testing.T) {
	cmd := quake.UserCmd{Forward: 400, Msec: 100}
	var events []netfault.Event
	var sent []Trace
	var applied []AppliedCommand
	for seq := uint32(1); seq <= 3; seq++ {
		v := seq
		action, stage := "forward", "after"
		if seq == 2 {
			action, stage = "drop", "blackout"
		} else {
			applied = append(applied, AppliedCommand{Connection: 1, Generation: 7, Sequence: seq, Kind: "new", Command: cmd})
		}
		events = append(events, netfault.Event{Direction: "client_to_server", Action: action, Stage: stage, Sequence: &v})
		sent = append(sent, Trace{Connection: 1, Generation: 7, Map: "base1", ClientSequence: seq, Command: cmd})
	}
	if r := VerifyNetworkCommands(events, sent, applied); !r.Accepted || r.Dropped != 1 || r.Forwarded.Matched != 2 {
		t.Fatal(r)
	}
	if r := VerifyNetworkCommands(events, sent, applied[:1]); r.Accepted {
		t.Fatal("missing forwarded command accepted")
	}
	bad := append([]AppliedCommand(nil), applied...)
	bad = append(bad, AppliedCommand{Connection: 1, Generation: 7, Sequence: 2, Kind: "new", Command: cmd})
	if r := VerifyNetworkCommands(events, sent, bad); r.Accepted {
		t.Fatal("dropped packet applied as new accepted")
	}
	bad = append([]AppliedCommand(nil), applied...)
	bad[1].Command.Forward = 0
	if r := VerifyNetworkCommands(events, sent, bad); r.Accepted {
		t.Fatal("changed command accepted")
	}
	if r := VerifyNetworkCommands(events[1:], sent, applied); r.Accepted {
		t.Fatal("missing proxy packet accepted")
	}
	if r := VerifyNetworkCommands(events, append(sent, sent[1]), applied); r.Accepted {
		t.Fatal("duplicate dropped trace command accepted")
	}
	// Recovery traffic is counted but cannot satisfy the missing new command.
	bad = append([]AppliedCommand(nil), applied...)
	bad[1].Kind = "old"
	if r := VerifyNetworkCommands(events, sent, bad); r.Accepted || r.Forwarded.RecoveryCommands != 1 {
		t.Fatal(r)
	}
}
