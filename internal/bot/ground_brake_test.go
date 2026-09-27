package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestGroundCoastBrake(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("BSP assets required")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	s := quake.Snapshot{Self: quake.Vec3{-34, -224, 24.125}, Health: 100, OnGround: true, SelfVelocity: quake.Vec3{-300, 0, 0}}
	cmd := quake.UserCmd{Msec: 100, Buttons: 1, Yaw: 1234, Pitch: 2345}
	got, ok := brakeGroundCoast(s, cmd, &g)
	if !ok {
		t.Fatalf("wall coast not braked: %+v", diagnoseGroundStep(s, cmd, &g))
	}
	if got.Buttons != cmd.Buttons || got.Yaw != cmd.Yaw || got.Pitch != cmd.Pitch {
		t.Fatal("brake changed aim/fire")
	}
	for _, kind := range []string{"air", "dead", "moving", "jump", "duck", "slow", "short", "friend", "mover", "observed_mover", "missing", "open"} {
		t.Run(kind, func(t *testing.T) {
			state, c, geo := s, cmd, g
			switch kind {
			case "air":
				state.OnGround = false
			case "dead":
				state.Health = 0
			case "moving":
				c.Forward = 80
			case "jump":
				c.Up = 400
			case "duck":
				state.Ducked = true
			case "slow":
				state.SelfVelocity[0] = -40
			case "short":
				c.Msec = 50
			case "friend":
				p := state.Self
				state.Teammate = &p
			case "mover":
				geo.Models = append([]quake.BSPModel(nil), g.Models...)
				geo.Models = append(geo.Models, quake.BSPModel{Min: state.Self, Max: state.Self})
			case "observed_mover":
				geo.Models = append([]quake.BSPModel(nil), g.Models...)
				far := quake.Vec3{state.Self[0] + 1000, state.Self[1], state.Self[2]}
				state.Movers = []quake.Mover{{Model: len(geo.Models), Origin: quake.Vec3{-1000, 0, 0}}}
				geo.Models = append(geo.Models, quake.BSPModel{Min: far, Max: far})
			case "missing":
				geo = quake.MapInfo{}
			case "open":
				state.Self[0] = 32
			}
			if out, accepted := brakeGroundCoast(state, c, &geo); accepted || out != c {
				t.Fatalf("unsupported brake %s: %+v", kind, out)
			}
		})
	}
}
