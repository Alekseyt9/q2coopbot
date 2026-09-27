package bot

import (
	"os"
	"path/filepath"
	"q2coopbot/internal/quake"
	"testing"
)

func TestBase3PickupNeedsNavigation(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("BSP/AAS required")
	}
	g, e := quake.LoadMap(root, "base3")
	if e != nil {
		t.Fatal(e)
	}
	n, e := quake.LoadAAS(filepath.Join(root, "maps/base3.aas"))
	if e != nil {
		t.Fatal(e)
	}
	p := &Planner{World: World{Geometry: &g, Goal: "cover_teammate"}}
	mate := quake.Vec3{882.625, 715.375, -807.875}
	s := quake.Snapshot{Frame: 3529, Health: 97, OnGround: true, InventoryKnown: true, Self: quake.Vec3{916.875, 710.375, -807.875}, Teammate: &mate, Inventory: []quake.InventoryItem{{Name: "Blaster", Count: 1}}, Pickups: []quake.Object{{ID: 383, Class: "weapon_shotgun", Origin: quake.Vec3{1168, 752, -816.875}}, {ID: 76, Class: "ammo_shells", Origin: quake.Vec3{1168, 712, -816.875}}}}
	if _, ok := p.pickupGoal(s); ok {
		t.Fatal("pickup without navigation")
	}
	p.Nav = n
	if _, ok := p.pickupGoal(s); !ok || p.pickup.attempt.Class != "weapon_shotgun" {
		t.Fatalf("shotgun not selected: %+v", p.pickup)
	}
	s.Self = healthStand(s.Pickups[0].Origin)
	s.Frame++
	s.Inventory = append(s.Inventory, quake.InventoryItem{Name: "Shotgun", Count: 1}, quake.InventoryItem{Name: "Shells", Count: 10})
	p.pickupGoal(s)
	if p.World.Pickup.State != "confirmed" {
		t.Fatal("weapon not confirmed")
	}
	s.Frame += 11
	if _, ok := p.pickupGoal(s); !ok || p.pickup.attempt.Class != "ammo_shells" {
		t.Fatal("shells not selected after weapon")
	}
}
