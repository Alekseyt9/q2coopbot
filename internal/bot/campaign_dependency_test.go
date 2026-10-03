package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

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
