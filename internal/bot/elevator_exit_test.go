package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestElevatorCompletionRequiresClearHullAndLanding(t *testing.T) {
	model := quake.BSPModel{Min: quake.Vec3{-152, 1336, -192}, Max: quake.Vec3{-16, 1480, 128}}
	for _, tc := range []struct {
		name   string
		pos    quake.Vec3
		ground bool
		done   bool
	}{
		{"still_over_platform", quake.Vec3{-22, 1408, 24}, true, false},
		{"airborne_beyond_edge", quake.Vec3{20, 1408, 24}, false, false},
		{"railing_above_floor", quake.Vec3{40, 1408, 80}, true, false},
		{"landed_beyond_old_distance_limit", quake.Vec3{117, 1408, 24}, true, true},
	} {
		stages := []string{"landing_unconfirmed"}
		if tc.done {
			stages = append(stages, "landing_probe")
		}
		for _, stage := range stages {
			t.Run(tc.name+"/"+stage, func(t *testing.T) {
				p := &Planner{goalPoint: quake.Vec3{10, 1408, 24}, elevator: &elevatorRide{model: 1, toArea: 2, stage: stage, exit: quake.Vec3{-20, 1408, 24}}, route: make([]quake.Waypoint, 2)}
				p.World.Geometry = &quake.MapInfo{Models: []quake.BSPModel{{}, model}}
				p.World.Snapshot = quake.Snapshot{Self: tc.pos, OnGround: tc.ground, Movers: []quake.Mover{{Model: 1}}}
				p.elevatorCommand(quake.UserCmd{}, quake.Waypoint{Model: 1, ToArea: 2})
				if (p.World.Elevator == "completed") != tc.done {
					t.Fatalf("completion=%q", p.World.Elevator)
				}
			})
		}
	}
}
func TestElevatorExitStandingHeadroom(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires local BSP")
	}
	g, err := quake.LoadMap(root, "base2")
	if err != nil {
		t.Fatal(err)
	}
	p := &Planner{goalPoint: quake.Vec3{10, 1408, 24}}
	p.World.Geometry = &g
	for _, tc := range []struct {
		name         string
		z            float64
		ground, want bool
	}{
		{"clear_exit", -7.875, true, true}, {"ceiling", 92.125, true, false}, {"too_far_below_landing", -100, true, false}, {"airborne", -7.875, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := quake.Snapshot{Self: quake.Vec3{-82.75, 1408, tc.z}, OnGround: tc.ground}
			if tc.name == "ceiling" {
				s.Self[0] = -22.75
			}
			if got := p.elevatorExitCanStand(s, quake.Vec3{-20, 1408, 24}); got != tc.want {
				t.Fatalf("standing=%v want=%v", got, tc.want)
			}
		})
	}
}
