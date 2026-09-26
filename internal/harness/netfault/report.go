package netfault

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

type DirectionReport struct {
	Forwarded        int    `json:"forwarded"`
	Dropped          int    `json:"dropped"`
	SequencedBefore  int    `json:"sequenced_before"`
	SequencedDropped int    `json:"sequenced_dropped"`
	SequencedAfter   int    `json:"sequenced_after"`
	RecoveryMS       *int64 `json:"recovery_ms,omitempty"`
}
type Report struct {
	Accepted   bool                        `json:"accepted"`
	State      string                      `json:"state"`
	Reason     string                      `json:"reason,omitempty"`
	ProblemRow int                         `json:"problem_row,omitempty"`
	Directions map[string]*DirectionReport `json:"directions"`
}

// ReadEvents rejects missing required fields as well as unknown fields. Zero
// elapsed time and zero-byte UDP datagrams are valid and must not hide omission.
func ReadEvents(path string) ([]Event, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 4096), 1024*1024)
	var rows []Event
	for s.Scan() {
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(s.Bytes(), &fields); err != nil {
			return nil, fmt.Errorf("row %d: %w", len(rows)+1, err)
		}
		for _, name := range []string{"elapsed_ms", "direction", "action", "bytes", "stage"} {
			if v, ok := fields[name]; !ok || bytes.Equal(v, []byte("null")) {
				return nil, fmt.Errorf("row %d: missing %s", len(rows)+1, name)
			}
		}
		var e Event
		d := json.NewDecoder(bytes.NewReader(s.Bytes()))
		d.DisallowUnknownFields()
		if err := d.Decode(&e); err != nil {
			return nil, fmt.Errorf("row %d: %w", len(rows)+1, err)
		}
		if d.Decode(new(any)) != io.EOF {
			return nil, fmt.Errorf("row %d: trailing data", len(rows)+1)
		}
		rows = append(rows, e)
	}
	return rows, s.Err()
}

// Analyze proves a sequenced transport blackout and recovery, not decoded game
// state, command application or companion behavior. Recovery is measured from
// the configured end of blackout to the first forwarded sequenced packet.
func Analyze(c Config, rows []Event, recoveryMS int, requireUpstreamLoss bool) Report {
	r := Report{State: "trace_invalid", Directions: map[string]*DirectionReport{"client_to_server": {}, "server_to_client": {}}}
	if err := c.Validate(); err != nil {
		r.Reason = err.Error()
		return r
	}
	if recoveryMS < 1 || recoveryMS > 120000 {
		r.Reason = "invalid recovery_ms"
		return r
	}
	armed := false
	last := int64(0)
	for i, e := range rows {
		invalid := func(reason string) Report { r.Reason = reason; r.ProblemRow = i + 1; return r }
		d, ok := r.Directions[e.Direction]
		if !ok {
			return invalid("invalid direction")
		}
		if e.ElapsedMS < last || e.ElapsedMS < 0 || e.Bytes < 0 || e.Bytes > 65507 {
			return invalid("invalid elapsed time or packet size")
		}
		if e.Sequence != nil && (e.Bytes < 4 || *e.Sequence > 0x7fffffff) {
			return invalid("invalid sequence metadata")
		}
		if e.Stage == "unarmed" {
			if armed || c.ArmBarrierDir == "" && e.Direction != "client_to_server" || e.ElapsedMS != 0 || e.Action != "forward" || e.Barrier != nil {
				return invalid("invalid unarmed packet")
			}
		} else {
			if !armed {
				if e.Direction != "server_to_client" || e.ElapsedMS != 0 {
					return invalid("missing first server packet")
				}
				if c.ArmBarrierDir != "" {
					if err := verifyBarrier(c, e); err != nil {
						return invalid(err.Error())
					}
				} else if e.Barrier != nil {
					return invalid("unexpected readiness evidence")
				}
				armed = true
			} else if e.Barrier != nil {
				return invalid("repeated readiness evidence")
			}
			// Compare milliseconds directly to avoid overflow on corrupted input.
			want := "after"
			if e.ElapsedMS < int64(c.AfterMS) {
				want = "before"
			} else if e.ElapsedMS < int64(c.AfterMS+c.DurationMS) {
				want = "blackout"
			}
			if e.Stage != want {
				return invalid("stage does not match configured interval")
			}
		}
		last = e.ElapsedMS
		wantAction := "forward"
		if e.Stage == "blackout" {
			wantAction = "drop"
		}
		if e.Action != wantAction {
			return invalid("action does not match stage")
		}
		if e.Action == "drop" {
			d.Dropped++
		} else {
			d.Forwarded++
		}
		if e.Sequence != nil {
			switch e.Stage {
			case "before":
				d.SequencedBefore++
			case "blackout":
				d.SequencedDropped++
			case "after":
				d.SequencedAfter++
				if d.RecoveryMS == nil {
					delay := e.ElapsedMS - int64(c.AfterMS+c.DurationMS)
					d.RecoveryMS = &delay
				}
			}
		}
	}
	r.State = "fixture_failed"
	for _, direction := range []string{"server_to_client", "client_to_server"} {
		d := r.Directions[direction]
		if d.SequencedBefore == 0 {
			r.Reason = direction + ": no sequenced traffic before blackout"
			return r
		}
		if (direction == "server_to_client" || requireUpstreamLoss) && d.SequencedDropped == 0 {
			r.Reason = direction + ": loss not exercised"
			return r
		}
		if d.RecoveryMS == nil {
			r.Reason = direction + ": recovery not observed"
			return r
		}
		if *d.RecoveryMS > int64(recoveryMS) {
			r.Reason = direction + ": recovery deadline exceeded"
			return r
		}
	}
	r.Accepted = true
	r.State = "passed"
	return r
}
