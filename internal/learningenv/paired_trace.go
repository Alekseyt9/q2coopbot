package learningenv

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"q2coopbot/internal/harness"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

// ReadPairedTrace validates the observation/capture envelope without inference.
// A paired reset currently requires one fresh connection per participant.
func ReadPairedTrace(r io.Reader) ([]harness.Trace, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 65536), 8*1024*1024)
	var rows []harness.Trace
	terminalSeen := false
	for scanner.Scan() {
		var row struct {
			harness.Trace
			ObservationFrame int             `json:"observation_frame"`
			Capture          *policy.Capture `json:"combat_policy"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			return nil, err
		}
		if terminalSeen {
			return nil, fmt.Errorf("paired trace continues after final observation")
		}
		if row.Capture == nil {
			return nil, fmt.Errorf("paired trace lacks capture")
		}
		c := row.Capture
		id := c.Observation.Identity
		if row.Health != nil && *row.Health != c.Observation.Health || row.Self != nil && *row.Self != c.Observation.Position {
			return nil, fmt.Errorf("paired trace state differs from capture observation")
		}
		if c.Observation.Version != policy.ObservationVersion || c.AppliedCommand != row.Command ||
			c.Applied.Version != policy.ActionVersion || c.Applied.Identity != id || id.Life < 1 ||
			id.Connection != 1 || row.Connection != id.Connection || id.Spawncount != row.Generation ||
			id.Actor != row.SelfEntity || id.Frame != row.ObservationFrame || id.Frame != row.Frame || id.Map != row.Map ||
			row.ClientSequence == 0 && !row.TerminalObservationOnly || c.ClientSequence != 0 && c.ClientSequence != row.ClientSequence {
			return nil, fmt.Errorf("paired trace observation/command identity mismatch")
		}
		if row.TerminalObservationOnly {
			if row.ClientSequence != 0 || c.ClientSequence != 0 || c.Provider != "terminal_observer" || row.Health == nil || row.Self == nil || c.Selection != nil || row.Command != (quake.UserCmd{}) {
				return nil, fmt.Errorf("invalid paired final observation envelope")
			}
			terminalSeen = true
		}
		rows = append(rows, row.Trace)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("empty paired trace")
	}
	return rows, nil
}

// BindTraces validates ALL native commands against BOTH complete client traces
// before exposing the selected actor to the existing transition assembler.
func (n *NativePairs) BindTraces(traces [2][]harness.Trace, role int) (*NativeSteps, []harness.AppliedCommand, error) {
	if role < 0 || role > 1 || n.Release == nil || len(n.Pairs) == 0 {
		return nil, nil, fmt.Errorf("invalid paired actor selection")
	}
	for r, rows := range traces {
		if len(rows) == len(n.Pairs)+1 {
			last := rows[len(rows)-1]
			end := n.Pairs[len(n.Pairs)-1].Steps[r]
			if !last.TerminalObservationOnly || last.Frame != end.EndFrame || last.Generation != end.Spawncount || last.SelfEntity != end.Actor || last.Connection != 1 {
				return nil, nil, fmt.Errorf("unmatched paired final observation")
			}
			rows = rows[:len(n.Pairs)]
		} else if len(rows) != len(n.Pairs) {
			return nil, nil, fmt.Errorf("paired role %d native/full-trace counts differ", r)
		}
		for i, row := range rows {
			p := n.Pairs[i].Steps[r]
			c := n.Pairs[i].Commands[r]
			if row.TerminalObservationOnly || row.Connection != 1 || row.Generation != p.Spawncount || row.Frame != p.BeginFrame ||
				row.ClientSequence != p.Sequence || row.SelfEntity != p.Actor || row.Command != c.Command {
				return nil, nil, fmt.Errorf("paired role %d trace command %d differs from native", r, i)
			}
		}
	}
	native := &NativeSteps{Release: n.Release}
	commands := make([]harness.AppliedCommand, 0, len(n.Pairs))
	for _, pair := range n.Pairs {
		native.Steps = append(native.Steps, pair.Steps[role])
		commands = append(commands, pair.Commands[role])
	}
	return native, commands, nil
}
