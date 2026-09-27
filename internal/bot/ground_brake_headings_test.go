package bot

import (
	"encoding/json"
	"math"
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestGroundBrakeAcrossViewDirections(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("BSP assets required")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	for _, east := range []bool{false, true} {
		name := "base1-edge-coast-brake.json"
		if east {
			name = "base1-edge-east-coast-brake.json"
		}
		data, err := os.ReadFile("../../scripts/scenarios/" + name)
		if err != nil {
			t.Fatal(err)
		}
		var fixture struct {
			Bot   quake.Vec3 `json:"bot_origin"`
			Actor quake.Vec3 `json:"actor_origin"`
		}
		if err = json.Unmarshal(data, &fixture); err != nil {
			t.Fatal(err)
		}
		vx := -300.
		if east {
			vx = 300
		}
		s := quake.Snapshot{Self: fixture.Bot, Teammate: &fixture.Actor, Health: 100, OnGround: true, SelfVelocity: quake.Vec3{vx, 0, 0}}
		s.Self[0] += vx * .1
		friend := quake.Snapshot{Self: fixture.Actor}
		visible, known := g.PointPVS(s.EyePoint(), friend.EyePoint())
		if !visible || !known || !g.ClearShot(s.EyePoint(), friend.EyePoint()) || g.GroundFrictionStatus(friend.Self) != "dry_flat" || !g.PlayerMoveClear(friend.Self, friend.Self) {
			t.Fatal("invalid scene actor", name)
		}
		for _, yaw := range []int16{0, 8192, 16384, 24576, -32768, -16384} {
			for _, pitch := range []int16{-12000, 0, 12000} {
				s.DeltaAngles = [3]int16{1000, 7000, 0}
				cmd := quake.UserCmd{Msec: 100, Yaw: yaw, Pitch: pitch, Buttons: 1, Impulse: 10}
				got, ok := brakeGroundCoast(s, cmd, &g)
				if !ok {
					t.Fatalf("missing brake east=%v yaw=%v pitch=%v", east, yaw, pitch)
				}
				if got.Yaw != cmd.Yaw || got.Pitch != cmd.Pitch || got.Buttons != cmd.Buttons || got.Impulse != cmd.Impulse {
					t.Fatal("aim/action changed")
				}
				p := predictGroundStep(s, got)
				if math.Hypot(p.Velocity[0], p.Velocity[1]) > 1 {
					t.Fatalf("view-space residual %+v", p)
				}
			}
		}
	}
}
