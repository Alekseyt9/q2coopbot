package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func routeGeometry(destinations ...string) *quake.MapInfo {
	m := &quake.MapInfo{Models: []quake.BSPModel{{}, {Min: quake.Vec3{100, 0, -24}, Max: quake.Vec3{160, 64, 64}}}}
	for _, destination := range destinations {
		m.Entities = append(m.Entities, quake.MapEntity{Class: "trigger_changelevel", Model: 1, Map: destination})
	}
	return m
}

func TestCampaignRouteProgressRequiresSelectedNativeMap(t *testing.T) {
	p := &Planner{Campaign: true, CampaignRoute: []string{"base1", "base2", "base3"}, World: World{Geometry: routeGeometry("base2$base1")}}
	s := quake.Snapshot{Map: "base1", Frame: 100, Health: 60, OnGround: true}
	goal, ok := p.campaignGoal(s)
	if !ok {
		t.Fatal(p.World.Campaign)
	}
	s.Self = goal
	p.campaignGoal(s)
	if p.campaignRouteIndex != 0 {
		t.Fatal("coordinates advanced route")
	}
	s.Map = "base3"
	if _, ok = p.campaignGoal(s); ok || p.campaignRouteIndex != 0 || p.World.Campaign.State != "unexpected_map_change" {
		t.Fatal("skipped map accepted")
	}
	s.Map = "base2"
	p.World.Geometry = routeGeometry("base1", "base3$base2")
	if _, ok = p.campaignGoal(s); !ok || p.campaignRouteIndex != 1 || p.campaignDestination != "base3" || p.World.Campaign.CompletedLevels != 1 {
		t.Fatal(p.World.Campaign)
	}
	for i := 0; i < 3; i++ {
		p.campaignGoal(s)
	}
	if p.campaignRouteIndex != 1 {
		t.Fatal("repeated snapshot advanced route")
	}
	s.Map = "base1"
	if _, ok = p.campaignGoal(s); ok || p.campaignRouteIndex != 1 {
		t.Fatal("unplanned return accepted")
	}
	s.Map = "base3"
	if _, ok = p.campaignGoal(s); ok || p.World.Campaign.State != "campaign_completed" || p.campaignRouteIndex != 2 {
		t.Fatal(p.World.Campaign)
	}
	p.campaignGoal(s)
	if p.World.Campaign.State != "campaign_completed" {
		t.Fatal("terminal progress lost")
	}
	q := &Planner{Campaign: true, CampaignRoute: []string{"base1", "base2"}}
	if _, ok = q.campaignGoal(quake.Snapshot{Map: "base2"}); ok || q.campaignRouteIndex != 0 {
		t.Fatal("starting at destination invented completion")
	}
}

func TestCampaignRouteMapResetAndCheckpoint(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base1/base2 BSP/AAS")
	}
	route := []string{"base1", "base2", "base3"}
	p := &Planner{Campaign: true, CampaignRoute: route, GameClock: true}
	s := quake.Snapshot{Map: "base1", Frame: 100, Health: 60, OnGround: true, Self: quake.Vec3{-1488, 1800, -23.875}}
	p.update(s, root)
	p.resources = map[int]*ResourceMemory{7: {Item: quake.Object{ID: 7, Class: "ammo_shells"}, LastSeen: 100}}
	p.pickup = &pickupTask{}
	p.deathPoint = &s.Self
	s.Map = "base2"
	s.Frame = 5
	s.Self = quake.Vec3{0, 0, 24}
	p.update(s, root)
	if p.World.GeometryStatus != "ready" || !p.World.AASLoaded || p.campaignDestination != "base3" || p.campaignRouteIndex != 1 || len(p.resources) != 0 || p.pickup != nil || p.deathPoint != nil || p.exitPreparation != nil {
		t.Fatal("map state leaked", p.World.Campaign)
	}
	state, err := p.CaptureCheckpoint()
	if err != nil {
		t.Fatal(err)
	}
	q := &Planner{Campaign: true, CampaignRoute: append([]string(nil), route...)}
	if err = q.RestoreCheckpoint(state, s, root); err != nil {
		t.Fatal(err)
	}
	if _, ok := q.campaignGoal(s); !ok || q.campaignDestination != "base3" || q.campaignRouteIndex != 1 {
		t.Fatal(q.World.Campaign)
	}
	wrong := &Planner{Campaign: true, CampaignRoute: []string{"base1", "base2", "base4"}}
	if wrong.RestoreCheckpoint(state, s, root) == nil {
		t.Fatal("different route accepted")
	}
	state.Campaign.RouteIndex = 0
	if state.validate(s.Map) == nil {
		t.Fatal("checkpoint map/progress mismatch accepted")
	}
}

func TestCampaignRouteValidation(t *testing.T) {
	for _, route := range [][]string{{"base1"}, {"base1", "base1"}, {"base1", "../base2"}, make([]string, 65)} {
		if validateCampaignRoute(route) == nil {
			t.Fatal(route)
		}
	}
	if validateCampaignRoute([]string{"base1", "base2", "base1"}) != nil {
		t.Fatal("explicit revisit refused")
	}
}

func TestBase2CampaignEntersGraphAfterNativeTransitionSpawn(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base2 BSP/AAS")
	}
	p := &Planner{Campaign: true, CampaignRoute: []string{"base1", "base2", "base3"}, campaignMap: "base1", campaignDestination: "base2", GameClock: true}
	s := quake.Snapshot{Map: "base2", Frame: 1, Health: 100, OnGround: false, Self: quake.Vec3{848, 2292, -214}}
	p.update(s, root)
	if p.routeOK {
		t.Fatal("airborne graph entry accepted")
	}
	s.Frame, s.OnGround, s.Self[2] = 10, true, -231.875
	p.update(s, root)
	if p.World.Goal != "reach_level_exit" || !p.routeOK || len(p.route) == 0 {
		t.Fatalf("spawn cannot enter campaign route: %s %s", p.World.Goal, p.World.Navigation)
	}
	if p.World.Campaign.Exit.Destination != "base3$base2b" {
		t.Fatal("suspended closer exit chosen", p.World.Campaign.Exit)
	}
	for _, exit := range p.World.Geometry.Exits() {
		if exit.Destination == "base3$base2a" {
			if _, ok := p.campaignExitContact(exit, s.Self); ok {
				t.Fatal("floor below suspended trigger accepted as contact")
			}
		}
	}
}
