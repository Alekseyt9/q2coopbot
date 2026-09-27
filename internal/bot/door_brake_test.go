package bot

import (
	"math"
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestClosingDoorCoastBrake(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("BSP assets required")
	}
	g, err := quake.LoadMap(root, "base2")
	if err != nil {
		t.Fatal(err)
	}
	s := quake.Snapshot{Map: "base2", Frame: 91, Health: 100, OnGround: true, Self: quake.Vec3{26, -800, 8.125}, SelfVelocity: quake.Vec3{300, 0, 0}, Movers: []quake.Mover{{ID: 50, Model: 27, Origin: quake.Vec3{0, 0, 52}}}}
	prev := s
	prev.Frame--
	prev.Self[0] = -4
	prev.Movers = []quake.Mover{{ID: 50, Model: 27, Origin: quake.Vec3{0, 0, 62}}}
	cmd := quake.UserCmd{Msec: 100, Yaw: 1000, Pitch: 2000, Buttons: 1, Impulse: 2}
	out, ok := brakeClosingDoorCoast(prev, s, cmd, &g)
	if !ok {
		t.Fatal("closing door coast was not braked")
	}
	p := predictGroundStep(s, out)
	if math.Hypot(p.Velocity[0], p.Velocity[1]) > 1 {
		t.Fatal("residual velocity")
	}
	unchanged := out
	unchanged.Forward = cmd.Forward
	unchanged.Side = cmd.Side
	if unchanged != cmd {
		t.Fatal("aim or buttons changed")
	}
	t.Run("one_tick_closure", func(t *testing.T) {
		state, old := s, prev
		state.Movers = []quake.Mover{{ID: 50, Model: 27, Origin: quake.Vec3{0, 0, 62}}}
		old.Movers = []quake.Mover{{ID: 50, Model: 27, Origin: quake.Vec3{0, 0, 72}}}
		active := cmd
		active.Forward = 400
		got, ok := brakeClosingDoorApproach(old, state, active, &g)
		if !ok || math.Hypot(predictGroundStep(state, got).Velocity[0], predictGroundStep(state, got).Velocity[1]) > 1 {
			t.Fatal("one-tick closure not arrested")
		}
		old.Frame--
		if _, ok := brakeClosingDoorApproach(old, state, active, &g); ok {
			t.Fatal("forecast across gap")
		}
		old.Frame++
		old.Movers[0].Origin[2] = 52
		if _, ok := brakeClosingDoorApproach(old, state, active, &g); ok {
			t.Fatal("opening door forecast")
		}
		old.Movers[0].Origin[2] = 72
		state.Self[0] = -4
		if _, ok := brakeClosingDoorApproach(old, state, active, &g); ok {
			t.Fatal("safe approach cancelled")
		}
	})
	t.Run("active_move_unsafe_but_neutral_coast_safe", func(t *testing.T) {
		state, old := s, prev
		state.Self[0] = 16
		state.SelfVelocity[0] = 160
		state.Movers = []quake.Mover{{ID: 50, Model: 27, Origin: quake.Vec3{0, 0, 62}}}
		old.Self[0] = 0
		old.Movers = []quake.Mover{{ID: 50, Model: 27, Origin: quake.Vec3{0, 0, 72}}}
		active := quake.UserCmd{Msec: 100, Forward: 160}
		if _, ok := brakeClosingDoorCoast(old, state, quake.UserCmd{Msec: 100}, &g); ok {
			t.Fatal("safe neutral coast should not need braking")
		}
		out, ok := brakeClosingDoorApproach(old, state, active, &g)
		if !ok || math.Hypot(predictGroundStep(state, out).Velocity[0], predictGroundStep(state, out).Velocity[1]) > 1 {
			t.Fatal("unsafe active command not arrested")
		}
	})
	for _, kind := range []string{"gap", "map", "unobserved", "opening", "horizontal", "identity", "overlap", "safe_coast", "air", "duck", "active", "friend", "short", "other_mover"} {
		t.Run(kind, func(t *testing.T) {
			state, old, c, geo := s, prev, cmd, g
			old.Movers = append([]quake.Mover(nil), prev.Movers...)
			switch kind {
			case "gap":
				old.Frame--
			case "map":
				old.Map = "base1"
			case "unobserved":
				old.Movers = nil
			case "opening":
				old.Movers[0].Origin[2] = 42
			case "horizontal":
				old.Movers[0].Origin[0] = 1
			case "identity":
				old.Movers[0].ID++
			case "overlap":
				state.Self[0] = 29
			case "safe_coast":
				state.Self[0] = -4
			case "air":
				state.OnGround = false
			case "duck":
				state.Ducked = true
			case "active":
				c.Forward = 100
			case "friend":
				point := state.Self
				state.Teammate = &point
			case "short":
				c.Msec = 50
			case "other_mover":
				geo.Models = append([]quake.BSPModel(nil), g.Models...)
				geo.Models = append(geo.Models, quake.BSPModel{Min: state.Self, Max: state.Self})
			}
			if got, accepted := brakeClosingDoorCoast(old, state, c, &geo); accepted || got != c {
				t.Fatal("unsupported brake accepted")
			}
		})
	}
}
