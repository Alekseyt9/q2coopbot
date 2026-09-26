package harness

import (
	"os"
	"path/filepath"
	"q2coopbot/internal/quake"
	"strings"
	"testing"
)

func TestAppliedCommandsRequireExactGenerationSequenceAndValues(t *testing.T) {
	cmd := quake.UserCmd{Forward: 100, Msec: 100}
	sent := []Trace{{Generation: 1, ClientSequence: 12, Command: cmd}, {Generation: 2, ClientSequence: 12, Command: cmd}}
	valid := []AppliedCommand{{Generation: 1, Sequence: 12, Kind: "new", Command: cmd}, {Generation: 2, Sequence: 12, Kind: "new", Command: cmd}}
	if r := VerifyAppliedCommands(sent, valid); !r.Accepted || r.Matched != 2 {
		t.Fatal(r)
	}
	for _, tc := range []struct {
		name   string
		mutate func([]AppliedCommand) []AppliedCommand
	}{
		{"missing", func(a []AppliedCommand) []AppliedCommand { return a[:1] }},
		{"generation", func(a []AppliedCommand) []AppliedCommand { a[1].Generation = 3; return a }},
		{"value", func(a []AppliedCommand) []AppliedCommand { a[1].Command.Forward = 99; return a }},
		{"duplicate", func(a []AppliedCommand) []AppliedCommand { return append(a, a[0]) }},
		{"recovery only", func(a []AppliedCommand) []AppliedCommand { a[1].Kind = "old"; return a }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := append([]AppliedCommand(nil), valid...)
			if r := VerifyAppliedCommands(sent, tc.mutate(a)); r.Accepted {
				t.Fatal(r)
			}
		})
	}
	if r := VerifyAppliedCommands(append(sent, sent[0]), valid); r.Accepted {
		t.Fatal("duplicate sent accepted")
	}
	if r := VerifyAppliedCommands(nil, nil); r.Accepted {
		t.Fatal("empty accepted")
	}
}

func TestAppliedLogRejectsMalformedAndOutOfRangeCommands(t *testing.T) {
	valid := "sv_test_applied_cmd spawncount=7 frame=42 seq=12 kind=new pitch=-32768 yaw=32767 roll=0 forward=100 side=0 up=0 buttons=1 impulse=0 msec=100 light=0"
	path := filepath.Join(t.TempDir(), "server.log")
	for _, line := range []string{valid, strings.Replace(valid, "32767", "32768", 1), strings.Replace(valid, "msec=100", "msec=256", 1), strings.Replace(valid, "kind=new", "kind=unknown", 1), valid + " extra", "sv_test_applied_cmd broken"} {
		if err := os.WriteFile(path, []byte("ordinary log\n"+line+"\n"), 0600); err != nil {
			t.Fatal(err)
		}
		a, err := ReadAppliedCommands(path)
		if line == valid {
			if err != nil || len(a) != 1 || a[0].Command.Pitch != -32768 {
				t.Fatal(a, err)
			}
		} else if err == nil {
			t.Fatal("malformed command accepted", line)
		}
	}
}
