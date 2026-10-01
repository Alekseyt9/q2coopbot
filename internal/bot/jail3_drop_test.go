package bot

import (
	"os"
	"path/filepath"
	"q2coopbot/internal/quake"
	"testing"
)

func TestJail3ShortDescent(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires jail3 BSP/AAS")
	}
	g, err := quake.LoadMap(root, "jail3")
	if err != nil {
		t.Fatal(err)
	}
	nav, err := quake.LoadAAS(filepath.Join(root, "maps/jail3.aas"))
	if err != nil {
		t.Fatal(err)
	}
	goal := quake.Vec3{-336.125, -680.125, 776.125}
	s := quake.Snapshot{Map: "jail3", Frame: 2054, Health: 7, OnGround: true, Self: quake.Vec3{-753.5, -648.75, 840.125}, Teammate: &goal}
	p := &Planner{Nav: nav, GameClock: true, World: World{Map: "jail3", Geometry: &g}}
	p.update(s, "")
	if !p.planShortWalkDown() {
		t.Fatal("captured descending staircase rejected")
	}
	if !p.jump.drop || p.jump.landing[2] >= s.Self[2]-24 || p.jump.landing[2] < s.Self[2]-40 {
		t.Fatalf("unexpected descent: %+v", p.jump)
	}
	// The front of the hull is already over the next, lower tread.
	front := p.jump.landing
	front[0] += 12
	front[1] += 12
	if _, flat := g.GroundDrop(front, 4); flat {
		t.Fatal("fixture no longer exercises descending stairs")
	}
	if _, supported := g.GroundDrop(front, 18.5); !supported {
		t.Fatal("next tread missing")
	}
	for _, name := range []string{"dead", "airborne", "lava", "slime", "no_route"} {
		t.Run(name, func(t *testing.T) {
			originalSnapshot, originalRoute := p.World.Snapshot, p.World.Route
			areas := append([]quake.Area(nil), nav.Areas...)
			defer func() { p.World.Snapshot, p.World.Route, nav.Areas = originalSnapshot, originalRoute, areas }()
			p.jump = nil
			switch name {
			case "dead":
				p.World.Snapshot.Health = 0
			case "airborne":
				p.World.Snapshot.OnGround = false
			case "no_route":
				p.World.Route = nil
			case "lava", "slime":
				flag := 2
				if name == "slime" {
					flag = 4
				}
				for i := range nav.Areas {
					nav.Areas[i].Contents |= flag
				}
			}
			if p.planShortWalkDown() {
				t.Fatal("unsafe descent accepted")
			}
		})
	}
}
