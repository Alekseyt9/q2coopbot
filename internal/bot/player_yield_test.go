package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestBase2PlayerPersonalSpace(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("assets")
	}
	g, err := quake.LoadMap(root, "base2")
	if err != nil {
		t.Fatal(err)
	}
	mate := quake.Vec3{400.125, -2401.75, -71.875}
	s := quake.Snapshot{Map: "base2", Frame: 4915, Health: 22, OnGround: true, Self: quake.Vec3{424.375, -2369.625, -71.875}, Teammate: &mate, TeammateEntity: 1,
		Movers: []quake.Mover{{Model: 17, Origin: quake.Vec3{0, 0, -66}}, {Model: 61, Origin: quake.Vec3{0, 0, -66}}}}
	p := &Planner{World: World{Geometry: &g, Goal: "cover_teammate"}}
	cmd, ok := p.playerYieldCommand(s, quake.UserCmd{})
	if !ok || cmd.Forward == 0 && cmd.Side == 0 || cmd.Up != 0 {
		t.Fatalf("blocked player: ok=%v cmd=%+v", ok, cmd)
	}
	for _, mode := range []string{"airborne", "dead", "different_floor", "comfortable", "missing_geometry"} {
		t.Run(mode, func(t *testing.T) {
			c := s
			q := *p
			q.World = p.World
			v := mate
			c.Teammate = &v
			switch mode {
			case "airborne":
				c.OnGround = false
			case "dead":
				c.Health = 0
			case "different_floor":
				v[2] += 48
			case "comfortable":
				v[1] -= 100
			case "missing_geometry":
				q.World.Geometry = nil
			}
			if _, ok := q.playerYieldCommand(c, quake.UserCmd{}); ok {
				t.Fatal("unsafe or unnecessary yielding")
			}
		})
	}
}
