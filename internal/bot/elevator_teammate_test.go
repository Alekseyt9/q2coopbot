package bot

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestElevatorTeammateExitWait(t *testing.T) {
	model := quake.BSPModel{Origin: quake.Vec3{0, 0, 0}}
	for _, tc := range []struct {
		name   string
		mate   *quake.Vec3
		target quake.Vec3
		ground bool
		moverZ float64
		want   bool
	}{
		{"blocked", &quake.Vec3{10, 0, 24}, quake.Vec3{53, 0, 24}, true, 0, true},
		{"released", &quake.Vec3{100, 0, 24}, quake.Vec3{53, 0, 24}, true, 0, false},
		{"alongside", &quake.Vec3{10, 70, 24}, quake.Vec3{53, 0, 24}, true, 0, false},
		{"above", &quake.Vec3{10, 0, 100}, quake.Vec3{53, 0, 24}, true, 0, false},
		{"hidden", nil, quake.Vec3{53, 0, 24}, true, 0, false},
		{"moving_away", &quake.Vec3{10, 0, 24}, quake.Vec3{-80, 0, 24}, true, 0, false},
		{"still_ascending", &quake.Vec3{10, 0, 24}, quake.Vec3{53, 0, 24}, true, -20, false},
		{"airborne", &quake.Vec3{10, 0, 24}, quake.Vec3{53, 0, 24}, false, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := quake.Snapshot{Self: quake.Vec3{-22.125, 0, 24}, OnGround: tc.ground, Teammate: tc.mate}
			mover := quake.Mover{Origin: quake.Vec3{0, 0, tc.moverZ}}
			if got := elevatorTeammateBlocksExit(s, tc.target, model, mover); got != tc.want {
				t.Fatalf("blocked=%v", got)
			}
			if tc.want {
				p := &Planner{}
				cmd := p.elevatorExitMove(quake.UserCmd{Forward: 300, Side: 100, Up: -200}, s, tc.target, model, mover)
				if cmd.Forward != 0 || cmd.Side != 0 || cmd.Up != 0 || p.World.Elevator != "exit_teammate_wait" {
					t.Fatalf("not neutral: %+v", cmd)
				}
			}
		})
	}
}
