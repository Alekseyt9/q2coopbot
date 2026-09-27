package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestGroundPredictionPathIncludesStoppingTail(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("BSP assets required")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	s := quake.Snapshot{Self: quake.Vec3{-34, -224, 24.125}, Health: 100, OnGround: true, SelfVelocity: quake.Vec3{-300, 0, 0}}
	p := diagnoseGroundStep(s, quake.UserCmd{Msec: 100}, &g)
	if p.CommandPath != "static_sampled_clear" || p.NeutralPath != "static_hull_blocked" || p.CommandThenStopPath != "static_hull_blocked" {
		t.Fatalf("missed inertial wall contact: %+v", p)
	}
	p = diagnoseGroundStep(s, quake.UserCmd{Msec: 100, Forward: 160}, &g)
	if p.CommandThenStopPath != "static_sampled_clear" || p.NeutralPath != "static_hull_blocked" {
		t.Fatalf("reverse command not distinguished from coasting: %+v", p)
	}
	p = diagnoseGroundStep(s, quake.UserCmd{Msec: 100}, nil)
	if p.CommandPath != "unknown_geometry" {
		t.Fatalf("unknown geometry accepted: %+v", p)
	}
}
