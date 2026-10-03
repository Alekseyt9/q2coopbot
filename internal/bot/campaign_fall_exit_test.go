package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

func TestBase2DeepExitStartsVerifiedDropAtEntry(t *testing.T) {
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
	s := quake.Snapshot{Map: "base2", Frame: 1000, Health: 51, Gravity: 800, OnGround: true, Self: quake.Vec3{-881.5, 142.125, -127.875}, Movers: []quake.Mover{
		{Model: 46, Origin: quake.Vec3{-4, 0, 0}}, {Model: 12},
		{Model: 47, Origin: quake.Vec3{-888, 180, -156}, Angles: quake.Vec3{0, 0, 270}},
		{Model: 48, Origin: quake.Vec3{-888, 12, -156}, Angles: quake.Vec3{0, 0, 90}},
	}}
	route := []quake.Waypoint{{Kind: 7, ToArea: 1971, Position: quake.Vec3{-888, 137, -128}}, {Kind: 7, ToArea: 1971, Position: quake.Vec3{-888, 135, -376}}, {Kind: 2, ToArea: 1988, Position: quake.Vec3{-892, 128.92929077148438, -384.0707092285156}}}
	now := time.Now()
	p := &Planner{Campaign: true, Nav: n, hasGoal: true, goalPoint: quake.Vec3{-884, 96, -407.875}, World: World{Geometry: &g, Snapshot: s, Map: s.Map, Goal: "reach_level_exit", Navigation: "ready", Route: route, Updated: now}}
	cmd := p.commandAt(quake.UserCmd{}, now)
	if p.jump == nil || !p.jump.drop || p.World.Command.LimitReason != "drop_prepare" {
		t.Fatalf("deep exit stopped: cmd=%+v decision=%+v jump=%+v", cmd, p.World.Command, p.jump)
	}
}

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
	if !p.planCornerDetour(s) || len(p.cornerDetour) < 2 {
		t.Fatal("no bounded grounded path around raised flap")
	}
	at := s.Self
	for _, waypoint := range p.cornerDetour {
		if _, ok := p.cornerGroundStep(s, at, waypoint); !ok {
			t.Fatal("detour contains an unsafe segment", at, waypoint)
		}
		at = waypoint
	}
	if !p.cornerWalkOffFrom(s, at) {
		t.Fatal("detour did not end at a verified descent")
	}
	s.Health = 10
	if p.cornerWalkOffFrom(s, at) {
		t.Fatal("dangerous descent accepted at low health")
	}
	s.Health = 100
	path := append([]quake.Vec3(nil), p.cornerDetour...)
	observed := s
	observed.Self = path[0]
	observed.Self[2] += 4 // Native hull rests on the adjacent upper tread.
	if _, ok := p.cornerDetourCommand(observed, quake.UserCmd{}); !ok || len(p.cornerDetour) >= len(path) {
		t.Fatal("upper-tread observation did not advance the detour")
	}
	p.cornerDetour = path
	if _, ok := p.cornerGroundStep(s, quake.Vec3{-803.75, 141.75, -131.875}, quake.Vec3{-811.75, 135.75, -127.875}); !ok {
		t.Fatal("native stair edge support rejected")
	}
	// A pose change invalidates the pending step rather than executing a
	// path computed for an old bridge orientation.
	s.Movers[2].Origin = p.cornerDetour[0]
	s.Movers[2].Angles = quake.Vec3{}
	cmd, _ := p.cornerDetourCommand(s, quake.UserCmd{})
	if cmd.Forward != 0 || cmd.Side != 0 || len(p.cornerDetour) != 0 {
		t.Fatal("changed mover pose did not invalidate the detour")
	}
}
