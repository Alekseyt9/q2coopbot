package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestBase3StairOppositeMomentum(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("assets")
	}
	g, err := quake.LoadMap(root, "base3")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(root + "/maps/base3.aas")
	if err != nil {
		t.Fatal(err)
	}
	s := quake.Snapshot{Map: "base3", Frame: 234, Health: 100, OnGround: true, Self: quake.Vec3{83, 585.75, -695.875}, SelfVelocity: quake.Vec3{-11.125, 73.625, 0}}
	goal := quake.Vec3{-148.375, -802.875, -231.875}
	p := &Planner{Nav: n, World: World{Map: s.Map, Geometry: &g, Snapshot: s, Goal: "regroup_after_respawn"}, goalPoint: goal}
	p.World.Route, _ = n.Route(s.Self, goal)
	if !p.planRampJump() {
		t.Fatal("no safe stair landing")
	}
	if !p.jump.steerVelocity || p.jump.phase != 4 || p.jump.landing[1] >= 540 || p.jump.landing[2] <= -690 {
		t.Fatalf("missing controlled upward landing: %+v", p.jump)
	}
	cmd, _ := p.jumpCommand(quake.UserCmd{})
	if cmd.Up != 0 || cmd.Forward != 60 || cmd.Yaw >= 0 {
		t.Fatalf("opposite momentum needs grounded uphill alignment: %+v", cmd)
	}
	p.World.Snapshot.Frame++
	p.World.Snapshot.Self = quake.Vec3{83, 579.125, -695.875}
	p.World.Snapshot.SelfVelocity = quake.Vec3{0, -60, 0}
	cmd, _ = p.jumpCommand(quake.UserCmd{})
	if p.jump == nil || p.jump.phase != 2 || cmd.Up != 200 || !p.jump.steerVelocity {
		t.Fatalf("alignment must revalidate takeoff: %+v %+v", p.jump, cmd)
	}
	// Correction must follow the observed velocity, including after reversal.
	p.jump.airborne = true
	p.World.Snapshot.OnGround = false
	p.World.Snapshot.Self = quake.Vec3{80, 530, -662.875}
	p.World.Snapshot.SelfVelocity = quake.Vec3{0, -350, -50}
	cmd, _ = p.jumpCommand(quake.UserCmd{})
	if cmd.Up != 0 || cmd.Yaw <= 0 {
		t.Fatalf("airborne overshoot must request downhill braking: %+v", cmd)
	}
}
