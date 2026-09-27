package bot

import (
	"math"
	"q2coopbot/internal/quake"
	"testing"
)

func TestGroundPredictionShortCommandsStopFully(t *testing.T) {
	s := quake.Snapshot{Health: 100, OnGround: true, SelfVelocity: quake.Vec3{300, 0, 0}}
	p := predictGroundStep(s, quake.UserCmd{Msec: 1})
	if p == nil || p.NeutralStopDistance < 40 || p.NeutralStopDistance > 43 {
		t.Fatalf("short timestep truncated stopping distance: %+v", p)
	}
}

func TestGroundPredictionRetainsMomentum(t *testing.T) {
	s := quake.Snapshot{Health: 100, OnGround: true, SelfVelocity: quake.Vec3{300, 0, 0}}
	for _, tc := range []struct {
		name string
		cmd  quake.UserCmd
		want quake.Vec3
	}{
		{"neutral", quake.UserCmd{Msec: 100}, quake.Vec3{12, 0, 0}},
		{"slow_reverse", quake.UserCmd{Msec: 100, Forward: -80}, quake.Vec3{4, 0, 0}},
		{"urgent_reverse", quake.UserCmd{Msec: 100, Forward: -160}, quake.Vec3{-4, 0, 0}},
		{"side", quake.UserCmd{Msec: 100, Side: 80}, quake.Vec3{12, -8, 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := predictGroundStep(s, tc.cmd)
			if p == nil {
				t.Fatal("missing prediction")
			}
			for i := range tc.want {
				if math.Abs(p.Displacement[i]-tc.want[i]) > 1e-6 {
					t.Fatalf("got%v want%v", p.Displacement, tc.want)
				}
			}
			if math.Abs(p.NeutralStopDistance-16.8) > 1e-6 {
				t.Fatalf("stop=%v", p.NeutralStopDistance)
			}
		})
	}
}

func TestGroundPredictionClampsSpeedAndUsesView(t *testing.T) {
	s := quake.Snapshot{Health: 100, OnGround: true, DeltaAngles: [3]int16{0, 16384, 0}}
	p := predictGroundStep(s, quake.UserCmd{Msec: 100, Forward: 400})
	if math.Abs(p.Displacement[0]) > 1e-6 || math.Abs(p.Displacement[1]-30) > 1e-6 {
		t.Fatal(p)
	}
	s.Ducked = true
	p = predictGroundStep(s, quake.UserCmd{Msec: 100, Forward: 400})
	if math.Abs(p.Displacement[1]-10) > 1e-6 {
		t.Fatal(p)
	}
	s.OnGround = false
	if predictGroundStep(s, quake.UserCmd{Msec: 100}) != nil {
		t.Fatal("airborne assumption")
	}
	s.OnGround = true
	s.SelfVelocity[2] = 80
	if predictGroundStep(s, quake.UserCmd{Msec: 100}) != nil {
		t.Fatal("vertical assumption")
	}
	s.SelfVelocity = quake.Vec3{}
	for _, cmd := range []quake.UserCmd{{}, {Msec: 101}, {Msec: 100, Up: 400}} {
		if predictGroundStep(s, cmd) != nil {
			t.Fatal("unsupported command")
		}
	}
}
