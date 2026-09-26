package harness

import "q2coopbot/internal/harness/netfault"

type NetworkCommandProof struct {
	Accepted  bool         `json:"accepted"`
	Reason    string       `json:"reason,omitempty"`
	Sent      int          `json:"sent"`
	Dropped   int          `json:"dropped"`
	Forwarded CommandProof `json:"forwarded_commands"`
}

// VerifyNetworkCommands permits absence of a new command only when its packet
// was explicitly dropped by the proxy. Recovery commands retain the existing
// separate accounting; they never substitute for a forwarded new command.
func VerifyNetworkCommands(events []netfault.Event, sent []Trace, applied []AppliedCommand) NetworkCommandProof {
	r := NetworkCommandProof{Sent: len(sent)}
	if len(sent) == 0 {
		r.Reason = "empty command trace"
		return r
	}
	packets := map[uint32]netfault.Event{}
	for _, e := range events {
		if e.Direction != "client_to_server" || e.Sequence == nil {
			continue
		}
		if _, ok := packets[*e.Sequence]; ok {
			r.Reason = "duplicate proxy client sequence"
			return r
		}
		packets[*e.Sequence] = e
	}
	seen := map[uint32]bool{}
	var forwarded []Trace
	for _, s := range sent {
		if s.Connection != 1 || s.Generation != sent[0].Generation || s.Map != sent[0].Map {
			r.Reason = "network command proof requires one fresh connection and generation"
			return r
		}
		if s.ClientSequence == 0 || seen[s.ClientSequence] {
			r.Reason = "missing or duplicate sent sequence"
			return r
		}
		seen[s.ClientSequence] = true
		e, ok := packets[s.ClientSequence]
		if !ok {
			r.Reason = "sent command missing from proxy log"
			return r
		}
		switch e.Action {
		case "forward":
			if e.Stage == "blackout" {
				r.Reason = "forward inside blackout"
				return r
			}
			forwarded = append(forwarded, s)
		case "drop":
			if e.Stage != "blackout" {
				r.Reason = "command dropped outside blackout"
				return r
			}
			r.Dropped++
		default:
			r.Reason = "invalid proxy command action"
			return r
		}
	}
	r.Forwarded = VerifyAppliedCommands(forwarded, applied)
	r.Accepted = r.Forwarded.Accepted
	if !r.Accepted {
		r.Reason = r.Forwarded.Reason
	}
	return r
}
