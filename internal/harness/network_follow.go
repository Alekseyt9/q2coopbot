package harness

import (
	"math"
	"q2coopbot/internal/harness/netfault"
	"q2coopbot/internal/quake"
)

type NetworkFollowReport struct {
	Accepted        bool   `json:"accepted"`
	State           string `json:"state"`
	Reason          string `json:"reason,omitempty"`
	ActorMovedFrame int    `json:"actor_moved_frame,omitempty"`
	ObservedFrame   int    `json:"observed_frame,omitempty"`
	FollowFrame     int    `json:"follow_frame,omitempty"`
	MovementFrame   int    `json:"movement_frame,omitempty"`
}

func horizontal(a, b quake.Vec3) float64 { return math.Hypot(a[0]-b[0], a[1]-b[1]) }

// VerifyNetworkFollow requires actual actor displacement during the proven
// blackout, a fresh observation of that actor, then a follow command and bot
// displacement. Scripted placement is allowed; this does not prove natural
// actor navigation or server application of every outgoing command.
func VerifyNetworkFollow(events []netfault.Event, bot, actor []Trace, budget int) NetworkFollowReport {
	r := NetworkFollowReport{State: "fixture_failed"}
	if budget < 1 || budget > 1000 {
		r.Reason = "invalid follow recovery budget"
		return r
	}
	proof := VerifyNetworkFrames(events, bot)
	if !proof.Accepted {
		r.State = proof.State
		r.Reason = proof.Reason
		return r
	}
	recovered := *proof.FirstRecoveredFrame
	beforeIndex := -1
	for i, row := range bot {
		if row.Frame == recovered {
			beforeIndex = i - 1
			break
		}
	}
	if beforeIndex < 0 {
		r.Reason = "missing pre-blackout bot frame"
		return r
	}
	before := bot[beforeIndex]
	end := recovered + budget
	actors := map[int]Trace{}
	for _, row := range actor {
		if row.Frame < before.Frame || row.Frame > end {
			continue
		}
		if _, exists := actors[row.Frame]; exists {
			r.State = "trace_invalid"
			r.Reason = "duplicate actor frame"
			return r
		}
		actors[row.Frame] = row
	}
	old, ok := actors[before.Frame]
	if !ok || old.Self == nil || old.SelfEntity <= 0 || old.Connection <= 0 {
		r.Reason = "missing actor identity before blackout"
		return r
	}
	if before.Health == nil || *before.Health <= 0 || before.Teammate == nil || before.TeammateEntity != old.SelfEntity || quake.Distance(*before.Teammate, *old.Self) > 16 {
		r.Reason = "actor not observed before blackout"
		return r
	}
	for frame := before.Frame; frame <= end; frame++ {
		a, exists := actors[frame]
		if !exists || a.Map != before.Map || a.Generation != before.Generation || a.Connection != old.Connection || a.SelfEntity != old.SelfEntity || a.Self == nil || a.Health == nil || *a.Health <= 0 {
			r.State = "trace_invalid"
			r.Reason = "incomplete or invalid actor recovery window"
			return r
		}
		if frame > before.Frame && frame < recovered && horizontal(*a.Self, *old.Self) >= 64 && r.ActorMovedFrame == 0 {
			r.ActorMovedFrame = frame
		}
	}
	if r.ActorMovedFrame == 0 {
		r.Reason = "actor did not move during blackout"
		return r
	}
	if bot[len(bot)-1].Frame < end {
		r.State = "trace_invalid"
		r.Reason = "incomplete bot recovery window"
		return r
	}
	r.State = "behavior_failed"
	var followOrigin *quake.Vec3
	for _, b := range bot {
		if b.Frame < recovered || b.Frame > end {
			continue
		}
		if b.Health == nil || *b.Health <= 0 || b.Self == nil {
			r.Reason = "bot not alive during recovery"
			return r
		}
		a := actors[b.Frame]
		fresh := b.Teammate != nil && b.TeammateEntity == old.SelfEntity && b.TeammateAgeFrames != nil && *b.TeammateAgeFrames == 0 && quake.Distance(*b.Teammate, *a.Self) <= 16 && horizontal(*a.Self, *old.Self) >= 64
		if fresh && r.ObservedFrame == 0 {
			r.ObservedFrame = b.Frame
		}
		if fresh && r.FollowFrame == 0 && b.Goal == "follow_teammate" && (b.Command.Forward != 0 || b.Command.Side != 0) {
			r.FollowFrame = b.Frame
			origin := *b.Self
			followOrigin = &origin
		}
		if followOrigin != nil && b.Frame > r.FollowFrame && horizontal(*b.Self, *followOrigin) >= 8 && r.MovementFrame == 0 {
			r.MovementFrame = b.Frame
		}
	}
	if r.ObservedFrame == 0 {
		r.Reason = "new actor position not observed before deadline"
		return r
	}
	if r.FollowFrame == 0 {
		r.Reason = "follow command not resumed before deadline"
		return r
	}
	if r.MovementFrame == 0 {
		r.Reason = "bot did not move after follow command"
		return r
	}
	r.Accepted = true
	r.State = "passed"
	return r
}
