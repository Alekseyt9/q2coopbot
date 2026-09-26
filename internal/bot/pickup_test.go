package bot

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestUsefulPickupUsesInventory(t *testing.T) {
	s := quake.Snapshot{InventoryKnown: true}
	if !usefulPickup(s, "weapon_machinegun") || usefulPickup(s, "ammo_bullets") {
		t.Fatal("new weapon/ammo ownership policy")
	}
	s.Inventory = []quake.InventoryItem{{Name: "Machinegun", Count: 1}, {Name: "Bullets", Count: 199}}
	if usefulPickup(s, "weapon_machinegun") || !usefulPickup(s, "ammo_bullets") {
		t.Fatal("owned weapon/needed ammo policy")
	}
	s.Inventory[1].Count = 200
	if usefulPickup(s, "ammo_bullets") {
		t.Fatal("full ammo still selected")
	}
	s.Inventory = append(s.Inventory, quake.InventoryItem{Name: "Body Armor", Count: 100})
	if usefulPickup(s, "item_armor_jacket") || usefulPickup(s, "item_armor_combat") || !usefulPickup(s, "item_armor_body") {
		t.Fatal("armor upgrade policy")
	}
	s.InventoryKnown = false
	if usefulPickup(s, "weapon_shotgun") {
		t.Fatal("unknown inventory treated as empty")
	}
	s.InventoryKnown = true
	s.InventoryAgeFrames = 21
	if usefulPickup(s, "weapon_shotgun") {
		t.Fatal("stale inventory accepted")
	}
}

func TestPickupRequiresInventoryGainAndReturnsControl(t *testing.T) {
	for _, confirm := range []bool{false, true} {
		goal := quake.Vec3{100, 0, 0}
		p := &Planner{World: World{Goal: "follow_teammate"}, pickup: &pickupTask{attempt: PickupAttempt{Class: "weapon_machinegun", Name: "Machinegun", Started: 10, State: "approach"}, progress: 10}}
		s := quake.Snapshot{Frame: 11, Health: 100, Teammate: &goal, InventoryKnown: true}
		if confirm {
			s.Inventory = []quake.InventoryItem{{Name: "Machinegun", Count: 1}}
		}
		p.pickupGoal(s)
		if !confirm {
			s.Frame = 16
			p.pickupGoal(s)
		}
		want := "unconfirmed"
		if confirm {
			want = "confirmed"
		}
		if p.pickup != nil || p.World.Pickup == nil || p.World.Pickup.State != want || p.pickupNext <= s.Frame {
			t.Fatalf("confirm=%v: %+v", confirm, p.World.Pickup)
		}
	}
}

func TestPickupStallIsBounded(t *testing.T) {
	goal := quake.Vec3{100, 0, 0}
	p := &Planner{World: World{Goal: "follow_teammate"}, pickup: &pickupTask{attempt: PickupAttempt{Entity: 5, Class: "ammo_bullets", Started: 10, State: "approach", Target: quake.Vec3{0, 0, 9.125}}, progress: 10}}
	s := quake.Snapshot{Frame: 35, Health: 100, Teammate: &goal, InventoryKnown: true, Pickups: []quake.Object{{ID: 5, Class: "ammo_bullets"}}}
	if _, ok := p.pickupGoal(s); ok || p.World.Pickup.State != "no_progress" {
		t.Fatalf("stalled pickup retained: %+v", p.World.Pickup)
	}
}
