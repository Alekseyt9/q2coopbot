package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestBunk1BridgeApproach(t *testing.T) {
	root, aas := os.Getenv("Q2_SEARCH_SCAN_ROOT"), os.Getenv("Q2_DROP_AAS")
	if root == "" || aas == "" {
		t.Skip("requires local bunk1 BSP/AAS")
	}
	g, err := quake.LoadMap(root, "bunk1")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(aas)
	if err != nil {
		t.Fatal(err)
	}
	goal := quake.Vec3{-524.875, 274.875, 16.125}
	p := &Planner{Nav: n, goalPoint: goal, World: World{Geometry: &g, Goal: "follow_teammate", Snapshot: quake.Snapshot{Self: quake.Vec3{-524.875, 166, 24.125}, OnGround: true, Health: 60, Movers: []quake.Mover{{Model: 99, Origin: quake.Vec3{0, -250, 0}}}}}}
	r, ok := p.bridgeRoute()
	if !ok || r[len(r)-1].Position[2] < 0 {
		t.Fatalf("route targets floor below bridge: %v", r)
	}
	if !p.bridgeNeedsApproach(goal) {
		t.Fatal("stopped before entering bridge")
	}
	cmd, ok := p.bridgeCommand(quake.UserCmd{Buttons: 1})
	if !ok || cmd.Forward == 0 || cmd.Up != 0 || cmd.Buttons != 0 {
		t.Fatalf("bad crossing: %+v", cmd)
	}
	p.World.Snapshot.Self = quake.Vec3{-524.875, 210, 16.125}
	if p.bridgeNeedsApproach(goal) {
		t.Fatal("cannot resume cover on bridge")
	}
	p.World.Snapshot.Self[2] = -167.875
	if _, ok := p.bridgeCommand(quake.UserCmd{}); ok {
		t.Fatal("crossing from floor under bridge")
	}
	p.World.Snapshot.Movers[0].Origin = quake.Vec3{}
	if _, _, ok := p.bridgeSupport(goal); ok {
		t.Fatal("closed bridge treated as support")
	}
	p.World.Snapshot.Movers = nil
	if _, _, ok := p.bridgeSupport(goal); ok {
		t.Fatal("hidden bridge treated as support")
	}
}
