package learningenv

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"

	"q2coopbot/internal/harness"
)

// AppendLog retains partial lines and reads only appended bytes. A missing
// file is normal before worker startup; truncation/replacement is an error.
type AppendLog struct {
	Offset  int64
	Partial []byte
	file os.FileInfo
}

func (r *AppendLog) Read(path string) ([]string, error) {
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if info.Size() < r.Offset {
		return nil, fmt.Errorf("live log truncated")
	}
	if r.file!=nil&&!os.SameFile(r.file,info){return nil,fmt.Errorf("live log replaced")};r.file=info
	if _, err = f.Seek(r.Offset, io.SeekStart); err != nil {
		return nil, err
	}
	var lines []string
	buffer := make([]byte, 65536)
	for {
		n, err := f.Read(buffer)
		if n > 0 {
			r.Offset += int64(n)
			r.Partial = append(r.Partial, buffer[:n]...)
			for {
				i := bytes.IndexByte(r.Partial, '\n')
				if i < 0 {
					break
				}
				lines = append(lines, strings.TrimSuffix(string(r.Partial[:i]), "\r"))
				r.Partial = r.Partial[i+1:]
			}
			if len(r.Partial) > 8*1024*1024 {
				return nil, fmt.Errorf("live log line too large")
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
	}
	return lines, nil
}

type liveKey struct {
	Generation int
	Sequence   uint32
}
type livePulse struct {
	Native  NativeStep
	Events  []DamageEvent
	Applied []harness.AppliedCommand
	EventOffset int
}
type LiveTelemetry struct {
	ClientName string
	Release    *CombatRelease
	ready      bool
	connection int
	active     []string
	commands   []harness.AppliedCommand
	last       *NativeStep
	pulses     map[liveKey]livePulse
	eventCount int
	eventStart int
}

func (l *LiveTelemetry) Push(line string) error {
	if strings.HasPrefix(line,"sv_test_damage "){l.eventCount++}
	if line == l.ClientName+" connected" {
		l.connection++
	}
	if line == "g_test_damage ready version=1" {
		l.ready = true
		return nil
	}
	if strings.HasPrefix(line, "sv_test_step ") {
		if strings.Contains(line, "phase=begin ") {
			if l.active != nil {
				return fmt.Errorf("overlapping live native pulses")
			}
			l.active = []string{line}
			l.commands = nil
			l.eventStart=l.eventCount
			return nil
		}
		if l.active == nil {
			return fmt.Errorf("native end/rejection outside live pulse")
		}
		l.active = append(l.active, line)
		if !l.ready {
			return fmt.Errorf("live damage telemetry not ready")
		}
		text := strings.Join(l.active, "\n") + "\n"
		events, err := ReadDamageEvents(strings.NewReader("g_test_damage ready version=1\n" + text))
		if err != nil {
			return err
		}
		native, err := ReadNativeWindow(strings.NewReader(text), events)
		if err != nil {
			return err
		}
		if len(native.Steps) != 1 {
			return fmt.Errorf("live pulse not unique")
		}
		p := native.Steps[0]
		if l.last != nil && (p.Spawncount != l.last.Spawncount || p.Actor != l.last.Actor || p.Sequence <= l.last.Sequence || p.BeginFrame != l.last.EndFrame) {
			return fmt.Errorf("live native discontinuity")
		}
		if native.Release != nil {
			if l.Release != nil {
				return fmt.Errorf("duplicate live release")
			}
			l.Release = native.Release
		}
		if l.pulses == nil {
			l.pulses = map[liveKey]livePulse{}
		}
		if len(l.pulses) > 10000 {
			return fmt.Errorf("live pulse cache too large")
		}
		l.pulses[liveKey{p.Spawncount, p.Sequence}] = livePulse{p, events, l.commands,l.eventStart}
		l.last = &p
		l.active = nil
		l.commands = nil
		return nil
	}
	if strings.HasPrefix(line, "sv_test_applied_cmd") {
		if l.active == nil {
			return fmt.Errorf("live dispatch outside pulse")
		}
		c, err := harness.ParseAppliedCommand(line, l.connection)
		if err != nil {
			return err
		}
		l.commands = append(l.commands, c)
	}
	if l.active != nil && (strings.HasPrefix(line, "sv_test_damage ") || strings.HasPrefix(line, "sv_test_combat ")) {
		l.active = append(l.active, line)
	}
	return nil
}

func (l *LiveTelemetry) Enrich(s *Step) (ServerOutcome, bool, error) {
	p, ok := l.pulses[liveKey{s.Observation.Identity.Spawncount, s.ClientSequence}]
	if !ok {
		return ServerOutcome{}, false, nil
	}
	if len(p.Applied) != 1 {
		return ServerOutcome{}, false, fmt.Errorf("live dispatch not unique")
	}
	n := NativeSteps{Steps: []NativeStep{p.Native}}
	matched, err := n.Match(s)
	if err != nil {
		return ServerOutcome{}, false, err
	}
	s.Native = matched
	execution := NewExecutionIndex(p.Applied).Match(s)
	s.Execution = &execution
	if !execution.Matched || s.Next != nil && !execution.WindowExclusive {
		return ServerOutcome{}, false, fmt.Errorf("live exact dispatch proof failed")
	}
	j := DamageJoiner{Events: p.Events}
	outcome:=j.JoinNative(s,matched)
	indexes:=append([]int{},matched.DamageIndexes...);for i:=range indexes{indexes[i]+=p.EventOffset};matched.DamageIndexes=indexes
	return outcome, true, nil
}
