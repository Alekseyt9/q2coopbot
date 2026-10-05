package learningenv

import "q2coopbot/internal/harness"

const ExecutionVersion = "server_dispatch_v1"

// Native telemetry is written after the command-time validator, immediately
// before calling ClientThink. It proves dispatch, not a hit or successful move.
type Execution struct {
	Version          string `json:"version"`
	Matched          bool   `json:"matched"`
	DispatchFrame    *int   `json:"dispatch_frame"`
	WindowExclusive  bool   `json:"window_exclusive"`
	RecoveryCommands int    `json:"recovery_commands"`
	Reason           string `json:"reason,omitempty"`
}

type executionKey struct {
	connection, spawncount int
	sequence               uint32
}
type ExecutionIndex struct {
	newCommands map[executionKey][]harness.AppliedCommand
	commands    []harness.AppliedCommand
}

func NewExecutionIndex(commands []harness.AppliedCommand) *ExecutionIndex {
	x := &ExecutionIndex{commands: commands, newCommands: map[executionKey][]harness.AppliedCommand{}}
	for _, c := range commands {
		if c.Kind == "new" {
			k := executionKey{c.Connection, c.Generation, c.Sequence}
			x.newCommands[k] = append(x.newCommands[k], c)
		}
	}
	return x
}

func (x *ExecutionIndex) Match(s *Step) Execution {
	e := Execution{Version: ExecutionVersion}
	id := s.Observation.Identity
	if s.ClientSequence == 0 || id.Connection <= 0 {
		e.Reason = "missing_sequence_or_connection"
		return e
	}
	commands := x.newCommands[executionKey{id.Connection, id.Spawncount, s.ClientSequence}]
	if len(commands) != 1 {
		e.Reason = "missing_or_duplicate_dispatch"
		return e
	}
	c := commands[0]
	if c.Command != s.Command {
		e.Reason = "dispatch_command_mismatch"
		return e
	}
	e.Matched = true
	e.DispatchFrame = &c.Frame
	if s.Next == nil || !sameWorld(s) || s.Next.Identity.Frame != id.Frame+1 {
		e.Reason = "no_consecutive_next_observation"
		return e
	}
	// Commands received after snapshot F and before the next tick execute at F.
	if c.Frame < id.Frame || c.Frame >= s.Next.Identity.Frame {
		e.Reason = "dispatch_outside_observation_window"
		return e
	}
	count := 0
	for _, other := range x.commands {
		if other.Connection == id.Connection && other.Generation == id.Spawncount && other.Frame >= id.Frame && other.Frame < s.Next.Identity.Frame {
			count++
			if other.Kind != "new" {
				e.RecoveryCommands++
			}
		}
	}
	e.WindowExclusive = count == 1
	if !e.WindowExclusive {
		e.Reason = "multiple_dispatches_in_window"
	}
	return e
}
