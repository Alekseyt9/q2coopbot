package harness

import (
	"q2coopbot/internal/harness/netfault"
	"q2coopbot/internal/quake"
	"testing"
)

func TestNetworkFollowNeedsFreshActorCommandAndMovement(t *testing.T) {
	hp := int16(100)
	age := 0
	seq := uint32(1)
	var events []netfault.Event
	var bot, actor []Trace
	for frame := 10; frame <= 17; frame++ {
		a := Trace{Map: "base1", Generation: 7, Connection: 1, Frame: frame, Health: &hp, SelfEntity: 1, Self: &quake.Vec3{160, 0, 0}}
		if frame >= 11 {
			a.Self = &quake.Vec3{320, 0, 0}
		}
		actor = append(actor, a)
		action, stage := "forward", "before"
		if frame == 11 || frame == 12 {
			action, stage = "drop", "blackout"
		} else if frame >= 13 {
			stage = "after"
		}
		events = append(events, netfault.Event{Direction: "server_to_client", Action: action, Stage: stage, Sequence: &seq, Frames: []netfault.GameFrame{{Map: "base1", Generation: 7, Frame: frame}}})
		if action == "drop" {
			continue
		}
		b := a
		b.Self = &quake.Vec3{0, 0, 0}
		b.SelfEntity = 2
		b.Teammate = a.Self
		b.TeammateEntity = 1
		b.TeammateAgeFrames = &age
		if frame >= 13 {
			b.Goal = "follow_teammate"
			b.Command.Forward = 400
			b.Self = &quake.Vec3{float64((frame - 13) * 10), 0, 0}
		}
		bot = append(bot, b)
	}
	if r := VerifyNetworkFollow(events, bot, actor, 4); !r.Accepted || r.FollowFrame != 13 || r.MovementFrame != 14 {
		t.Fatal(r)
	}
	for _, test := range []struct {
		name string
		edit func([]Trace, []Trace)
	}{
		{"stale observation", func(b, a []Trace) {
			old := 1
			for i := range b {
				b[i].TeammateAgeFrames = &old
			}
		}},
		{"wrong teammate", func(b, a []Trace) {
			for i := range b {
				b[i].TeammateEntity = 3
			}
		}},
		{"no command", func(b, a []Trace) {
			for i := range b {
				b[i].Command = quake.UserCmd{}
			}
		}},
		{"no movement", func(b, a []Trace) {
			for i := range b {
				b[i].Self = &quake.Vec3{}
			}
		}},
		{"stationary actor", func(b, a []Trace) {
			for i := range a {
				a[i].Self = &quake.Vec3{160, 0, 0}
			}
		}},
		{"actor trace gap", func(b, a []Trace) { a[2].Frame++ }},
	} {
		t.Run(test.name, func(t *testing.T) {
			b, a := append([]Trace(nil), bot...), append([]Trace(nil), actor...)
			test.edit(b, a)
			if r := VerifyNetworkFollow(events, b, a, 4); r.Accepted {
				t.Fatal(r)
			}
		})
	}
}
