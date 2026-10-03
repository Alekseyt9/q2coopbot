package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func localChainPlanner(t *testing.T) (*Planner, quake.Snapshot) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires BSP/AAS")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(root + "/maps/base1.aas")
	if err != nil {
		t.Fatal(err)
	}
	g.Entities = []quake.MapEntity{
		{Class: "func_door", Model: 32, Origin: quake.Vec3{1792, -1792, -104}, TargetName: "main", Wait: -1},
		{Class: "func_door", Model: 33, Origin: quake.Vec3{1872, -1888, -104}, TargetName: "side", Wait: -1},
		{Class: "trigger_once", Model: 16, Origin: quake.Vec3{120, 32, 0}, Target: "main"},
		{Class: "trigger_once", Model: 13, Origin: quake.Vec3{2160, -1888, -56}, Target: "side"},
	}
	s := quake.Snapshot{Map: "base1", Frame: 10, Self: quake.Vec3{135, -305.5, 24.125}, Health: 100, OnGround: true, Movers: []quake.Mover{{Model: 32, Origin: g.Entities[0].Origin}, {Model: 33, Origin: g.Entities[1].Origin}}}
	p := &Planner{Campaign: true, Nav: n, World: World{Map: "base1", Geometry: &g, Goal: "reach_level_exit", Campaign: &CampaignDecision{}, Snapshot: s}}
	return p, s
}

func TestLocalActivationNestingResumeAndCycleGuard(t *testing.T) {
	p, s := localChainPlanner(t)
	p.discoverCampaignDependency(s, -1, 0)
	parent := p.campaignDependency
	if parent == nil || parent.DoorModel != 32 {
		t.Fatal("root dependency absent")
	}
	s.Self = quake.Vec3{160, -295, 24.125}
	s.Frame = 20
	p.discoverCampaignDependency(s, 1, 0)
	child := p.campaignDependency
	if child == nil || child.Parent != parent || child.DoorModel != 33 || child.Activation.Trigger.Model != 13 {
		t.Fatal("blocked activator lost original goal", child)
	}
	p.discoverCampaignDependency(s, 1, 0)
	if p.campaignDependency != child {
		t.Fatal("duplicate/cyclic dependency")
	}
	s.Movers[1].Origin[2] -= 80
	s.Frame = 25
	p.campaignDependencyGoal(s)
	if p.campaignDependency != child {
		t.Fatal("partial opening still intersects corridor")
	}
	s.Movers[1].Origin[2] -= 48
	s.Frame = 30
	goal, ok, active := p.campaignDependencyGoal(s)
	if !ok || !active || p.campaignDependency != parent || goal != parent.Goal || parent.started != 20 {
		t.Fatal("parent not resumed or budget not paused", parent)
	}
	if len(p.campaignOpenedDoors) != 1 {
		t.Fatal("permanent opening not remembered")
	}
	// A later closed observation revokes memory and cannot modify UDP data.
	s.Movers[1].Origin[2] += 128
	p.campaignDoorMovers(s)
	if len(p.campaignOpenedDoors) != 0 {
		t.Fatal("contradiction retained open memory")
	}
	p.campaignDependency = child
	s.Frame = 400
	_, ok, active = p.campaignDependencyGoal(s)
	if ok || !active || child.State != "activation_timeout" || p.campaignDependency != child || child.Parent != parent {
		t.Fatal("failed child silently discarded")
	}
	p.World.Snapshot = s
	if _, err := p.CaptureCheckpoint(); err == nil {
		t.Fatal("nested local intention silently dropped on save")
	}
}

func TestCampaignPermanentDoorMemoryAndShortReplan(t *testing.T) {
	p, s := localChainPlanner(t)
	for _, m := range s.Movers {
		m.Origin[2] -= 128
		p.rememberCampaignDoor(m)
	}
	s.Self = quake.Vec3{215.75, -282.375, 24.125}
	s.Movers = nil
	goal := quake.Vec3{40, -304, 24.125}
	if route, ok := p.campaignOpenedRoute(s, goal); !ok || len(route) != 1 || route[0].Position != goal {
		t.Fatal("safe short passage not replanned")
	}
	if s.Movers != nil {
		t.Fatal("remembered poses fabricated native observations")
	}
	s.Movers = []quake.Mover{{Model: 33, Origin: quake.Vec3{1872, -1888, -104}}}
	if _, ok := p.campaignOpenedRoute(s, goal); ok {
		t.Fatal("closed observation ignored")
	}
	p.campaignOpenedDoors = nil
	p.World.Geometry.Entities[0].Wait = 3
	p.rememberCampaignDoor(quake.Mover{Model: 32})
	if len(p.campaignOpenedDoors) != 0 {
		t.Fatal("temporary door cached")
	}
	p.World.Geometry.Entities[0].Wait = -1
	p.World.Geometry.Entities[0].SpawnFlags = 32
	p.rememberCampaignDoor(quake.Mover{Model: 32})
	if len(p.campaignOpenedDoors) != 0 {
		t.Fatal("toggle door cached")
	}
}
