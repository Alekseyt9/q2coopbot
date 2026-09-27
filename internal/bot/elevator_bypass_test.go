package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestElevatorBypassValidatedAndFallback(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires BSP")
	}
	g, err := quake.LoadMap(root, "base2")
	if err != nil {
		t.Fatal(err)
	}
	model, _ := g.Model(50)
	mover := quake.Mover{Model: 50}
	mate := quake.Vec3{10, 1408, 24.125}
	s := quake.Snapshot{Self: quake.Vec3{-22.125, 1408, 24.125}, OnGround: true, Teammate: &mate, Movers: []quake.Mover{mover}}
	p := &Planner{elevator: &elevatorRide{}}
	p.World.Geometry = &g
	// The exact shared floor/platform edge must not invent a gap. A point
	// above the same edge must still be rejected as unsupported.
	if !elevatorSupportedPoint(&g, mover, quake.Vec3{0, 1408, 24.125}) || elevatorSupportedPoint(&g, mover, quake.Vec3{0, 1408, 80}) {
		t.Fatal("incorrect shared-edge support")
	}
	target := quake.Vec3{53, 1408, 24.125}
	cmd, ok := p.elevatorBypass(quake.UserCmd{}, s, target, model, mover)
	if !ok || cmd.Forward == 0 && cmd.Side == 0 || len(p.elevator.bypass) != 2 {
		t.Fatal("verified bypass not selected")
	}
	// Move the teammate into the chosen lateral segment. Revalidation must
	// cancel it rather than trusting the previously saved waypoints.
	mate = quake.Vec3{-22.125, 1428, 24.125}
	if _, ok = p.elevatorBypass(quake.UserCmd{}, s, target, model, mover); ok || len(p.elevator.bypass) != 0 {
		t.Fatal("changed blocker ignored")
	}
	p.World.Geometry = nil
	mate = quake.Vec3{10, 1408, 24.125}
	cmd = p.elevatorExitMove(quake.UserCmd{Forward: 300}, s, target, model, mover)
	if cmd.Forward != 0 || cmd.Side != 0 || p.World.Elevator != "exit_teammate_wait" {
		t.Fatal("unknown support must wait")
	}
	p.World.Geometry = &g
	mover.Origin[2] = -5
	if _, ok = p.elevatorBypass(quake.UserCmd{}, s, target, model, mover); ok {
		t.Fatal("moving platform accepted")
	}
}

func TestElevatorBlockedExitRetreat(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires BSP")
	}
	g, err := quake.LoadMap(root, "base2")
	if err != nil {
		t.Fatal(err)
	}
	model, _ := g.Model(50)
	mover := quake.Mover{Model: 50}
	mate := quake.Vec3{23.125, 1408, 24.125}
	s := quake.Snapshot{Self: quake.Vec3{-9, 1408, 24.125}, OnGround: true, Teammate: &mate, Movers: []quake.Mover{mover, {Model: 1, Origin: quake.Vec3{-576, -4, -24}}, {Model: 1, Origin: quake.Vec3{-576, 60, -24}}}}
	p := &Planner{elevator: &elevatorRide{}}
	p.World.Geometry = &g
	target := quake.Vec3{53, 1408, 24.125}
	cmd := p.elevatorExitMove(quake.UserCmd{}, s, target, model, mover)
	if p.World.Elevator != "exit_teammate_retreat" || p.elevator.waitAnchor == nil || cmd.Forward == 0 && cmd.Side == 0 {
		t.Fatal("did not retreat from crush zone")
	}
	s.Self = *p.elevator.waitAnchor
	cmd = p.elevatorExitMove(quake.UserCmd{}, s, target, model, mover)
	if p.World.Elevator != "exit_teammate_wait" || cmd.Forward != 0 || cmd.Side != 0 {
		t.Fatal("did not hold safe position")
	}
	// A teammate beside the exit must not keep the retreat latch set.
	mate = quake.Vec3{23.125, 1480, 24.125}
	cmd = p.elevatorExitMove(quake.UserCmd{}, s, target, model, mover)
	if p.elevator.waitAnchor != nil || cmd.Forward == 0 && cmd.Side == 0 {
		t.Fatal("did not release clear corridor")
	}
}
