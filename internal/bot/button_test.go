package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestCampaignButtonOnBlockedDirectRoute(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base2 BSP/AAS")
	}
	g, err := quake.LoadMap(root, "base2")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(root + "/maps/base2.aas")
	if err != nil {
		t.Fatal(err)
	}
	s := quake.Snapshot{Map: "base2", Frame: 20, Health: 100, OnGround: true, Self: quake.Vec3{194, 1971.875, -151.875}, Movers: []quake.Mover{{Model: 33}, {Model: 34}}}
	p := &Planner{Campaign: true, Nav: n, goalPoint: quake.Vec3{194, 2024, -151.875}, World: World{Geometry: &g, GeometryStatus: "ready", Goal: "reach_level_exit", Navigation: "direct_clear"}}
	task := p.selectButtonTask(s)
	if task == nil || task.doorModel != 33 || task.buttonModel != 34 || !task.campaign {
		t.Fatal("blocked direct route did not select original linked button", task)
	}
	s.Movers = s.Movers[:1]
	if p.selectButtonTask(s) != nil {
		t.Fatal("unobserved button selected")
	}
	s.Movers = []quake.Mover{{Model: 33, Origin: quake.Vec3{0, 0, 74}}, {Model: 34}}
	if p.selectButtonTask(s) != nil {
		t.Fatal("open passage selected button again")
	}
}

func TestButtonOwnerLossCancelsBeforeSearchEarlyReturn(t *testing.T) {
	for _, kind := range []string{"hidden", "changed", "dead"} {
		t.Run(kind, func(t *testing.T) {
			mate := quake.Vec3{200, 0, 24}
			p := &Planner{World: World{Map: "test", Geometry: &quake.MapInfo{}}, button: &buttonTask{teammateEntity: 1, started: 1}}
			s := quake.Snapshot{Map: "test", Frame: 10, Health: 100, Teammate: &mate, TeammateEntity: 1}
			switch kind {
			case "hidden":
				s.Teammate = nil
			case "changed":
				s.TeammateEntity = 2
			case "dead":
				s.Health = 0
			}
			p.update(s, "")
			if p.button != nil || p.buttonCooldown != 40 {
				t.Fatalf("old interaction survived: %+v", p.button)
			}
		})
	}
}

func TestButtonDoesNotOverrideNewParentGoal(t *testing.T) {
	for _, recover := range []bool{false, true} {
		mate := quake.Vec3{200, 0, 24}
		hp := int16(100)
		want := "cover_teammate"
		s := quake.Snapshot{Map: "test", Frame: 10, Self: quake.Vec3{150, 0, 24}, Teammate: &mate, TeammateEntity: 1, Health: hp}
		if recover {
			s.Health = 20
			s.Pickups = []quake.Object{{Class: "item_health", Origin: quake.Vec3{180, 0, 24}}}
			want = "recover_health"
		}
		p := &Planner{World: World{Map: "test", Geometry: &quake.MapInfo{}}, button: &buttonTask{teammateEntity: 1, started: 1}}
		p.update(s, "")
		if p.button != nil || p.World.Goal != want {
			t.Fatalf("recover=%t task=%+v goal=%s", recover, p.button, p.World.Goal)
		}
	}
}
