package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestCity3OriginalHighShootButton(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires city3 BSP/AAS")
	}
	g, err := quake.LoadMap(root, "city3")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(root + "/maps/city3.aas")
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range g.Entities {
		if e.Model == 70 || e.Model == 69 {
			t.Fatal("excluded difficulty door/button retained")
		}
	}
	s := quake.Snapshot{Map: "city3", Frame: 20, Health: 100, OnGround: true, Self: quake.Vec3{-330.25, -67, -55.875}, Movers: []quake.Mover{{Model: 81}, {Model: 82}}}
	p := &Planner{Campaign: true, Nav: n, goalPoint: quake.Vec3{-328, 0, -39.875}, World: World{Geometry: &g, GeometryStatus: "ready", Goal: "reach_level_exit", Navigation: "ready"}}
	task := p.selectButtonTask(s)
	if task == nil || task.buttonModel != 82 || task.doorModel != 81 || task.action != "shoot" || len(task.chain) != 1 {
		t.Fatal("exact original link not selected", task)
	}
}
