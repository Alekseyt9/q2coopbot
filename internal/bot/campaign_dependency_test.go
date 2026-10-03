package bot

import (
	"context"
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestCampaignWaypointRequiresIsolatedTestConfig(t *testing.T) {
	for _, cfg := range []Config{
		{TestCampaignGoal: "1,2,3"},
		{Campaign: true, FramePaced: true, Host: "192.168.1.1", TestTeleportMap: "base2", TestCampaignGoal: "1,2,3"},
		{Campaign: true, FramePaced: true, Host: "127.0.0.1", TestTeleportMap: "base2", TestCampaignGoal: "NaN,2,3"},
		{Campaign: true, FramePaced: true, Host: "127.0.0.1", TestTeleportMap: "base2", TestCampaignGoal: "1,2,3", CheckpointRestore: "save.json"},
	} {
		if err := Run(context.Background(), cfg); err == nil {
			t.Fatal("unsafe test waypoint configuration accepted")
		}
	}
}

func TestBase2ActivationDoorLip(t *testing.T) {
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
	from := quake.Vec3{676.75, 1787, 24.125}
	to := quake.Vec3{705.43783045, 1759.12512988, 24.125}
	for _, test := range []struct {
		name     string
		offset   float64
		released bool
	}{
		{"closed", 0, false}, {"partial", -80, false}, {"eight_unit_lip", -120, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			d := &CampaignDependency{DoorModel: 24, ProbeFrom: from, ProbeTo: to, started: 1, Goal: quake.Vec3{476, 1788, 24.125}}
			p := &Planner{Campaign: true, Nav: n, campaignDependency: d, World: World{Geometry: &g, Campaign: &CampaignDecision{}}}
			s := quake.Snapshot{Frame: 50, OnGround: true, Self: from, Movers: []quake.Mover{{Model: 24, Origin: quake.Vec3{0, 0, test.offset}}}}
			if test.released && g.MoverHullClear(s.Movers[0], from, to) {
				t.Fatal("fixture must require native step over lip")
			}
			if test.offset == 0 {
				if _, ok := p.campaignDependencyRoute(s, d.Goal); !ok {
					t.Fatal("safe flat route to original trigger unavailable")
				}
				if _, ok := p.checkedCampaignGroundRoute(s, quake.Vec3{768, 1792, 24.125}); ok {
					t.Fatal("flat fallback crossed closed door")
				}
			}
			_, _, active := p.campaignDependencyGoal(s)
			if active == test.released || (p.campaignDependency == nil) != test.released {
				t.Fatal("incorrect opening confirmation", active, p.campaignDependency)
			}
		})
	}
}

func TestBase2BSPCampaignDependency(t *testing.T) {
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
	s := quake.Snapshot{Map: "base2", Frame: 800, Health: 100, OnGround: true, Self: quake.Vec3{96, -213.25, 8.125}, Movers: []quake.Mover{{Model: 53}, {Model: 49, Origin: quake.Vec3{256, -800, -9}}, {Model: 15}, {Model: 18}, {Model: 8}, {Model: 27}}}
	p := &Planner{Campaign: true, CampaignRoute: []string{"base1", "base2", "base3"}, campaignRouteIndex: 1, Nav: n, World: World{Map: "base2", Geometry: &g, GeometryStatus: "ready", Goal: "reach_level_exit", Snapshot: s}}
	p.discoverCampaignDependency(s, 0, -1)
	if p.campaignDependency == nil || p.campaignDependency.Activation.Trigger.Model != 20 || len(p.campaignDependency.Activation.Chain) != 2 || p.campaignDependency.Activation.Chain[1].Model != 49 {
		t.Fatal("BSP chain not found", p.campaignDependency)
	}
	goal, ok := p.campaignGoal(s)
	if !ok || p.World.Campaign.State != "unlock_exit_route" {
		t.Fatal(p.World.Campaign)
	}
	if len(p.campaignDependency.UnitConditions) != 0 {
		t.Fatal("base2 flyer cross-level trigger falsely linked to door53")
	}
	route, ok := p.campaignDependencyRoute(s, goal)
	if ok || len(route) != 0 || p.campaignDependency.State != "activation_route_unavailable" {
		t.Fatal("route through locked door accepted")
	}
	s.Self = quake.Vec3{252, -808, 24.125}
	if _, ok = p.campaignDependencyRoute(s, goal); !ok {
		t.Fatal("reachable local activation rejected")
	}
	s.Movers[0].Origin[2] = 100
	if _, ok, active := p.campaignDependencyGoal(s); ok || active || p.campaignDependency != nil {
		t.Fatal("observed opening did not release dependency")
	}
	s.Movers = nil
	p.campaignDependency = nil
	p.discoverCampaignDependency(s, 0, -1)
	if p.campaignDependency != nil {
		t.Fatal("unobserved door generated activation goal")
	}
}
