package harness

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"q2coopbot/internal/quake"
)

type AppliedCommand struct {
	Generation int
	Frame      int
	Sequence   uint32
	Kind       string
	Command    quake.UserCmd
}

func ReadAppliedCommands(path string) ([]AppliedCommand, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var rows []AppliedCommand
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 4096), 4*1024*1024)
	line := 0
	for scanner.Scan() {
		line++
		text := scanner.Text()
		if !strings.HasPrefix(text, "sv_test_applied_cmd") {
			continue
		}
		var r AppliedCommand
		c := &r.Command
		n, err := fmt.Sscanf(text, "sv_test_applied_cmd spawncount=%d frame=%d seq=%d kind=%s pitch=%d yaw=%d roll=%d forward=%d side=%d up=%d buttons=%d impulse=%d msec=%d light=%d", &r.Generation, &r.Frame, &r.Sequence, &r.Kind, &c.Pitch, &c.Yaw, &c.Roll, &c.Forward, &c.Side, &c.Up, &c.Buttons, &c.Impulse, &c.Msec, &c.Light)
		if err != nil || n != 14 || len(strings.Fields(text)) != 15 || r.Generation < 0 || r.Frame < 0 || r.Sequence == 0 {
			return nil, fmt.Errorf("invalid applied command at server log line %d", line)
		}
		rows = append(rows, r)
	}
	return rows, scanner.Err()
}

type CommandProof struct {
	Accepted         bool   `json:"accepted"`
	Sent             int    `json:"sent"`
	AppliedNew       int    `json:"applied_new"`
	Matched          int    `json:"matched"`
	RecoveryCommands int    `json:"recovery_commands"`
	Reason           string `json:"reason,omitempty"`
	Generation       int    `json:"problem_generation,omitempty"`
	Sequence         uint32 `json:"problem_sequence,omitempty"`
}

// VerifyAppliedCommands matches the entire observer trace, including setup and
// tail, by generation and sequence. Server frame is not the observation frame.
func VerifyAppliedCommands(sent []Trace, applied []AppliedCommand) CommandProof {
	r := CommandProof{Sent: len(sent)}
	type key struct {
		generation int
		sequence   uint32
	}
	setProblem := func(reason string, k key) {
		if r.Reason == "" {
			r.Reason = reason
			r.Generation = k.generation
			r.Sequence = k.sequence
		}
	}
	sentKeys := map[key]quake.UserCmd{}
	for _, row := range sent {
		k := key{row.Generation, row.ClientSequence}
		if row.ClientSequence == 0 {
			setProblem("missing_client_sequence", k)
		}
		if _, ok := sentKeys[k]; ok {
			setProblem("duplicate_sent_sequence", k)
		}
		sentKeys[k] = row.Command
	}
	appliedKeys := map[key]quake.UserCmd{}
	for _, row := range applied {
		if row.Kind != "new" {
			r.RecoveryCommands++
			continue
		}
		r.AppliedNew++
		k := key{row.Generation, row.Sequence}
		if _, ok := appliedKeys[k]; ok {
			setProblem("duplicate_applied_sequence", k)
		}
		appliedKeys[k] = row.Command
		if _, ok := sentKeys[k]; !ok {
			setProblem("unexpected_applied_sequence", k)
		}
	}
	for _, row := range sent {
		k := key{row.Generation, row.ClientSequence}
		cmd, ok := appliedKeys[k]
		if !ok {
			setProblem("command_not_applied", k)
		} else if cmd != row.Command {
			setProblem("applied_command_mismatch", k)
		} else {
			r.Matched++
		}
	}
	if len(sent) == 0 {
		r.Reason = "empty_command_trace"
	}
	r.Accepted = r.Reason == "" && r.Sent == r.AppliedNew && r.Sent == r.Matched
	return r
}
