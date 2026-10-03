package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

func TestBase2CampaignEscapesUnclassifiedDuckedPosition(t *testing.T) {
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
	s := quake.Snapshot{Map: "base2", Frame: 3139, Health: 18, Gravity: 800, OnGround: true, Ducked: true, Self: quake.Vec3{55.75, 1088.625, 24.125}, Movers: []quake.Mover{{Model: 33}, {Model: 34}, {Model: 50, Origin: quake.Vec3{0, 0, -190.125}}}}
	goal := quake.Vec3{-132, -784, 8.125}
	if _, ok := n.Route(s.Self, goal); ok || g.PlayerMoveClear(s.Self, s.Self) || !g.CrouchMoveClear(s.Self, s.Self) {
		t.Fatal("fixture must be in a low ceiling outside standing AAS")
	}
	now := time.Now()
	p := &Planner{Campaign: true, Nav: n, hasGoal: true, goalPoint: goal, World: World{Map: s.Map, Geometry: &g, GeometryStatus: "ready", Goal: "reach_level_exit", Snapshot: s, Updated: now}}
	route, ok := p.regroupEntryRoute(s, goal)
	if !ok || len(route) < 2 {
		t.Fatal("no safe crouched entry")
	}
	if !g.CrouchStepClear(s.Self, route[0].Position[0]-s.Self[0], route[0].Position[1]-s.Self[1], quake.Horizontal(s.Self, route[0].Position)) {
		t.Fatal("entry is unsafe", route[0])
	}
	p.World.Route, p.World.Navigation = route, "ready"
	cmd := p.commandAt(quake.UserCmd{}, now)
	if cmd.Up >= 0 || cmd.Forward == 0 && cmd.Side == 0 {
		t.Fatal("did not crawl out", cmd, p.World.Command)
	}
	s.Ducked = false
	if _, ok := p.regroupEntryRoute(s, goal); ok {
		t.Fatal("standing body allowed through low ceiling")
	}
	s.Ducked = true
	s.OnGround = false
	if _, ok := p.regroupEntryRoute(s, goal); ok {
		t.Fatal("airborne entry accepted")
	}
}
