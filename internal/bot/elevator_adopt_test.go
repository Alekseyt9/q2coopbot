package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestElevatorAdoptsAlreadyReachedUpperLanding(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires BSP")
	}
	g, err := quake.LoadMap(root, "base2")
	if err != nil {
		t.Fatal(err)
	}
	model, _ := g.Model(50)
	p := &Planner{goalPoint: quake.Vec3{100, 1388, 24.125}}
	p.World.Geometry = &g
	for _, tc := range []struct {
		name   string
		self   quake.Vec3
		ground bool
		z      float64
		want   bool
	}{
		{"shared_ride", quake.Vec3{-77.25, 1427.25, 24.125}, true, 0, true},
		{"stuck_edge", quake.Vec3{-3, 1406.375, 24.125}, true, 0, true},
		{"lower_floor", quake.Vec3{-77, 1427, -165.875}, true, 0, false},
		{"ascending", quake.Vec3{-77, 1427, 14.125}, true, -10, false},
		{"airborne", quake.Vec3{-77, 1427, 24.125}, false, 0, false},
		{"outside", quake.Vec3{100, 1388, 24.125}, true, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := p.elevatorAtUpperLanding(quake.Snapshot{Self: tc.self, OnGround: tc.ground}, model, quake.Mover{Model: 50, Origin: quake.Vec3{0, 0, tc.z}}); got != tc.want {
				t.Fatalf("adopt=%v", got)
			}
		})
	}
}
