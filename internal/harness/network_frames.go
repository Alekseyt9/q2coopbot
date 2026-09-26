package harness

import (
	"fmt"
	"q2coopbot/internal/harness/netfault"
)

type NetworkFramesReport struct {
	Accepted               bool   `json:"accepted"`
	State                  string `json:"state"`
	Reason                 string `json:"reason,omitempty"`
	ProblemRow             int    `json:"problem_row,omitempty"`
	ObservedFrames         int    `json:"observed_frames"`
	ExplainedMissingFrames int    `json:"explained_missing_frames"`
	FirstRecoveredFrame    *int   `json:"first_recovered_frame,omitempty"`
}

// VerifyNetworkFrames attributes every missing frame inside the recorded bot
// window to a decoded datagram dropped by the proxy. Forwarded datagrams do
// not prove client reception: a missing forwarded frame remains a failure.
// This first fixture requires one map generation and one client connection.
func VerifyNetworkFrames(events []netfault.Event, rows []Trace) NetworkFramesReport {
	r := NetworkFramesReport{State: "trace_invalid"}
	if len(rows) < 2 {
		r.Reason = "insufficient bot frames"
		return r
	}
	type delivery struct {
		forwarded, dropped bool
		stage              string
	}
	packets := map[netfault.GameFrame]delivery{}
	for i, e := range events {
		if e.DecodeError != "" {
			r.Reason = "proxy decode error: " + e.DecodeError
			r.ProblemRow = i + 1
			return r
		}
		for _, f := range e.Frames {
			if e.Direction != "server_to_client" || e.Sequence == nil || f.Map == "" || f.Frame < 0 || (e.Action != "forward" && e.Action != "drop") {
				r.Reason = "invalid proxy frame metadata"
				r.ProblemRow = i + 1
				return r
			}
			p := packets[f]
			if e.Action == "drop" {
				if e.Stage != "blackout" {
					r.Reason = "frame dropped outside blackout"
					return r
				}
				p.dropped = true
			} else {
				p.forwarded = true
				p.stage = e.Stage
			}
			packets[f] = p
		}
	}
	first := rows[0]
	key := func(frame int) netfault.GameFrame {
		return netfault.GameFrame{Map: first.Map, Generation: first.Generation, Frame: frame}
	}
	for i, row := range rows {
		r.ProblemRow = i + 1
		if row.Connection <= 0 || row.Connection != first.Connection || row.Map != first.Map || row.Generation != first.Generation {
			r.Reason = "network fixture requires a single map generation and connection"
			return r
		}
		p := packets[key(row.Frame)]
		if !p.forwarded {
			r.Reason = "observed frame has no forwarded proxy packet"
			return r
		}
		if i > 0 {
			previous := rows[i-1].Frame
			if row.Frame <= previous {
				r.Reason = "bot frames do not increase"
				return r
			}
			// Bound work by the evidence size, including corrupted frame numbers.
			if row.Frame-previous-1 > len(packets) {
				r.Reason = "frame gap exceeds proxy evidence"
				return r
			}
			for frame := previous + 1; frame < row.Frame; frame++ {
				missing := packets[key(frame)]
				if !missing.dropped || missing.forwarded {
					r.Reason = fmt.Sprintf("unexplained missing frame %d", frame)
					return r
				}
				r.ExplainedMissingFrames++
			}
		}
		if r.ExplainedMissingFrames > 0 && p.stage == "after" && r.FirstRecoveredFrame == nil {
			frame := row.Frame
			r.FirstRecoveredFrame = &frame
		}
		r.ObservedFrames++
	}
	r.ProblemRow = 0
	if r.ExplainedMissingFrames == 0 || r.FirstRecoveredFrame == nil {
		r.State = "fixture_failed"
		r.Reason = "frame loss and recovery not exercised in bot window"
		return r
	}
	r.Accepted = true
	r.State = "passed"
	return r
}
