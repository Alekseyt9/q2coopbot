package bot

import (
	"encoding/json"
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestDirectionalEdgeFixtures(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("BSP assets required")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"north", "south", "diagonal"} {
		t.Run(name, func(t *testing.T) {
			data, err := os.ReadFile("../../scripts/scenarios/base1-edge-" + name + "-coast-brake.json")
			if err != nil {
				t.Fatal(err)
			}
			var f struct {
				Start    quake.Vec3 `json:"bot_origin"`
				Actor    quake.Vec3 `json:"actor_origin"`
				Stop     quake.Vec3 `json:"brake_origin"`
				Velocity quake.Vec3 `json:"brake_velocity"`
			}
			if err := json.Unmarshal(data, &f); err != nil {
				t.Fatal(err)
			}
			s := quake.Snapshot{Self: f.Stop, Health: 100, OnGround: true, SelfVelocity: f.Velocity, Teammate: &f.Actor}
			friend := quake.Snapshot{Self: f.Actor}
			vis, known := g.PointPVS(s.EyePoint(), friend.EyePoint())
			if !vis || !known || !g.ClearShot(s.EyePoint(), friend.EyePoint()) || g.GroundFrictionStatus(f.Actor) != "dry_flat" || !g.PlayerMoveClear(f.Actor, f.Actor) || g.GroundPathStatus(f.Start, f.Stop, false) != "static_sampled_clear" {
				t.Fatal("invalid fixture geometry/visibility")
			}
			for _, yaw := range []int16{0, 8192, 16384, -16384} {
				cmd := quake.UserCmd{Msec: 100, Yaw: yaw, Buttons: 1}
				prediction := diagnoseGroundStep(s, cmd, &g)
				if prediction == nil || prediction.NeutralPath != "uneven_or_missing_support" {
					t.Fatal("not an open edge", prediction)
				}
				got, ok := brakeGroundCoast(s, cmd, &g)
				if !ok || got.Yaw != cmd.Yaw || got.Buttons != cmd.Buttons {
					t.Fatal("directional brake failed", got, ok)
				}
			}
		})
	}
}
