package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestButtonShootSafetyAndEffect(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base2 BSP")
	}
	g, err := quake.LoadMap(root, "base2")
	if err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"fire", "settle", "explosive", "friend", "hidden", "blocked", "effect", "budget"} {
		t.Run(kind, func(t *testing.T) {
			s := quake.Snapshot{Map: "base2", Frame: 20, OnGround: true, Health: 100, Weapon: "Blaster", Self: quake.Vec3{320, 1940, -167.875}, Movers: []quake.Mover{{Model: 33}, {Model: 34}}}
			task := &buttonTask{campaign: true, action: "shoot", phase: "shoot", buttonModel: 34, doorModel: 33, aimSince: 10, started: 1}
			p := &Planner{button: task, World: World{Geometry: &g, GeometryStatus: "ready"}}
			from := s.EyePoint()
			from[2] -= 8
			target := quake.Vec3{320, 1979, -152}
			prev := quake.UserCmd{Yaw: quake.YawTo(from, target, 0), Pitch: quake.PitchTo(from, target, 0)}
			switch kind {
			case "settle":
				task.aimSince = 19
			case "explosive":
				s.Weapon = "Rocket Launcher"
			case "friend":
				mate := quake.Vec3{320, 1955, -167.875}
				s.Teammate = &mate
			case "hidden":
				s.Movers = s.Movers[:1]
			case "blocked":
				s.Movers[0].Origin = quake.Vec3{126, -32, 0}
			case "effect":
				s.Movers[1].Origin[1] = 4
				p.applyButtonTask(s)
				if task.phase != "wait_effect" {
					t.Fatal("observed movement did not stop shots")
				}
			case "budget":
				task.shots = 3
			}
			cmd, active := p.buttonShotCommand(s, prev, quake.UserCmd{Forward: 400, Buttons: 1})
			if !active || cmd.Forward != 0 || cmd.Side != 0 || cmd.Up != 0 {
				t.Fatal("shoot task did not hold position", cmd)
			}
			if (cmd.Buttons == 1) != (kind == "fire") {
				t.Fatal(kind, cmd, p.World.Command)
			}
			if kind == "fire" {
				s.Frame++
				cmd, _ = p.buttonShotCommand(s, cmd, cmd)
				if cmd.Buttons != 0 {
					t.Fatal("continuous button fire")
				}
			}
		})
	}
}
