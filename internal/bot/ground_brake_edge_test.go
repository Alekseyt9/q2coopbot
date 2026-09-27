package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestGroundCoastBrakeAtEdge(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("BSP assets required")
	}
	g, e := quake.LoadMap(root, "base1")
	if e != nil {
		t.Fatal(e)
	}
	s := quake.Snapshot{Self: quake.Vec3{460, 140, -39.875}, Health: 100, OnGround: true, SelfVelocity: quake.Vec3{-300, 0, 0}}
	cmd := quake.UserCmd{Msec: 100}
	friend := quake.Snapshot{Self: quake.Vec3{490, 280, -39.875}}
	if g.GroundFrictionStatus(friend.Self) != "dry_flat" || !g.PlayerMoveClear(friend.Self, friend.Self) || !g.ClearShot(s.EyePoint(), friend.EyePoint()) {
		t.Fatal("fixture teammate is not on a visible supported platform")
	}
	visible, known := g.PointPVS(s.EyePoint(), friend.EyePoint())
	if !visible || !known {
		t.Fatal("fixture lacks valid PVS")
	}
	s.Teammate = &friend.Self
	p := diagnoseGroundStep(s, cmd, &g)
	if p.NeutralPath != "uneven_or_missing_support" || !g.PlayerMoveClear(s.Self, quake.Vec3{443.2, 140, -39.875}) {
		t.Fatalf("fixture is not an open edge: %+v", p)
	}
	if got, ok := brakeGroundCoast(s, cmd, &g); !ok || got.Forward != 120 {
		t.Fatalf("edge not braked: %+v %v", got, ok)
	}
	for _, x := range []float64{440, 490} {
		s.Self[0] = x
		if _, ok := brakeGroundCoast(s, cmd, &g); ok {
			t.Fatalf("unsupported or safe position braked x=%v", x)
		}
	}
}
