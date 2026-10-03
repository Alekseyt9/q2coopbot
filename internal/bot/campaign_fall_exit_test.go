package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestBase2CampaignFallsThroughAlternateExit(t *testing.T) {
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
	s := quake.Snapshot{Map: "base2", Frame: 800, Health: 100, Gravity: 800, OnGround: true, Self: quake.Vec3{96, -213.25, 8.125}, Movers: []quake.Mover{{Model: 53}}}
	p := &Planner{Campaign: true, CampaignRoute: []string{"base1", "base2", "base3"}, campaignRouteIndex: 1, Nav: n, World: World{Map: "base2", Geometry: &g, Goal: "reach_level_exit"}}
	p.discoverCampaignDependency(s, 0, -1)
	goal, _ := p.campaignGoal(s)
	if _, ok := p.campaignDependencyRoute(s, goal); ok {
		t.Fatal("locked door route accepted")
	}
	goal, ok := p.campaignGoal(s)
	if !ok || p.campaignDependency != nil || p.campaignExitOverride != 38 || p.World.Campaign.Exit.Destination != "base3$base2a" || p.World.Campaign.ExitApproach != "fall_through_trigger" || goal[2] > -400 {
		t.Fatal("falling exit ignored", p.World.Campaign, goal)
	}
	if route, ok := p.campaignDependencyNavigator(s).Route(s.Self, goal); !ok || len(route) == 0 {
		t.Fatal("fallback not reachable without door53")
	}
	p.campaignExitOverride, p.campaignExitBlock = 0, nil
	p.discoverCampaignDependency(s, 0, -1)
	p.campaignDependency.State = "activation_route_unavailable"
	s.Health = 10
	p.campaignGoal(s)
	if p.campaignExitOverride != 0 {
		t.Fatal("lethal descent accepted")
	}
}

func TestBase2LowerExitCornerEscapesRaisedFlap(t *testing.T) {
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
	s := quake.Snapshot{Map: "base2", Frame: 900, Health: 100, Gravity: 800, OnGround: true, Self: quake.Vec3{-830.75, 167.5, -131.875}, Movers: []quake.Mover{
		{Model: 46, Origin: quake.Vec3{-4, 0, 0}}, {Model: 12},
		{Model: 47, Origin: quake.Vec3{-888, 180, -156}, Angles: quake.Vec3{0, 0, 270}},
		{Model: 48, Origin: quake.Vec3{-888, 12, -156}, Angles: quake.Vec3{0, 0, 90}},
	}}
	goal := quake.Vec3{-884, 96, -407.875}
	route, ok := n.Route(s.Self, goal)
	if !ok || len(route) == 0 {
		t.Fatal("missing exit route")
	}
	p := &Planner{Nav: n, goalPoint: goal, World: World{Map: "base2", Geometry: &g, Goal: "reach_level_exit", Route: route}}
	dx, dy, ok := p.regroupCornerStep(s, quake.Vec3{-838, 169, -128})
	if !ok {
		t.Fatal("all corner steps rejected")
	}
	end := s.Self
	end[0] += dx
	end[1] += dy
	if g.MoverHullClear(s.Movers[2], s.Self, end) || !g.MoverHullEscapeClear(s.Movers[2], s.Self, end) {
		t.Fatal("step did not escape the observed raised flap", dx, dy)
	}
}
