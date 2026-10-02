package bot

import (
	"os"
	"path/filepath"
	"testing"

	"q2coopbot/internal/quake"
)

func TestBase1ResourceDetourExposureAndRetry(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires local base1 BSP/AAS")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(filepath.Join(root, "maps/base1.aas"))
	if err != nil {
		t.Fatal(err)
	}
	item := quake.Object{ID: 308, Class: "weapon_shotgun", Origin: quake.Vec3{220, 1182, -240.875}}
	s := quake.Snapshot{Frame: 120, Health: 100, OnGround: true, InventoryKnown: true, Self: quake.Vec3{401, 1120, -231.875}}
	p := &Planner{Campaign: true, Nav: n, World: World{Geometry: &g, Goal: "reach_level_exit"}, resources: map[int]*ResourceMemory{308: {Item: item, State: "unknown", LastSeen: 30}}}
	for _, tc := range []struct {
		origin  quake.Vec3
		reason  string
		allowed bool
	}{
		{quake.Vec3{224, 898, -231.875}, "no_route_exposure", true},
		{quake.Vec3{296, 898, -231.875}, "route_exposure", false},
		{quake.Vec3{344, 1042, -231.875}, "current_exposure", false},
	} {
		s.Enemies = []quake.Object{{ID: 400, Class: "monster_soldier", Origin: tc.origin}}
		if got := p.resourceDetourAllowed(s, item, healthStand(item.Origin), true); got != tc.allowed || p.World.ResourceRisk.Reason != tc.reason {
			t.Fatalf("origin=%v allowed=%v risk=%+v", tc.origin, got, p.World.ResourceRisk)
		}
	}
	// A close visible weapon remains useful under fire; memory trips do not.
	near := s.Self
	near[0] -= 40
	if !p.resourceDetourAllowed(s, item, near, false) || p.resourceDetourAllowed(s, item, near, true) {
		t.Fatal("near visible pickup and optional memory trip conflated")
	}
	if _, ok := p.pickupGoal(s); ok || p.resources[item.ID].Attempted {
		t.Fatal("unsafe acquisition started or consumed the memory visit")
	}
	s.Enemies = nil
	if _, ok := p.pickupGoal(s); !ok || !p.pickup.attempt.FromMemory {
		t.Fatal("safe memory return not selected")
	}
	s.Frame++
	s.Enemies = []quake.Object{{ID: 400, Class: "monster_soldier", Origin: quake.Vec3{344, 1042, -231.875}}}
	if _, ok := p.pickupGoal(s); ok || p.pickup != nil || p.World.Pickup.State != "unsafe_route" || p.resources[item.ID].Attempted {
		t.Fatal("new threat did not cancel the visit while preserving retry")
	}
	s.Enemies = nil
	s.Frame += 10
	if _, ok := p.pickupGoal(s); !ok {
		t.Fatal("return did not recover after threat/cooldown")
	}
}
