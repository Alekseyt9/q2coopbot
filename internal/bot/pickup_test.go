package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestReturnPickupKeepsParentAndYieldsToPlayer(t *testing.T) {
	for _, meet := range []bool{false, true} {
		parent := &respawnRegroup{target: quake.Vec3{500, 0, 0}}
		p := &Planner{respawnRegroup: parent, World: World{Goal: "regroup_after_respawn"}, pickup: &pickupTask{duringReturn: true, attempt: PickupAttempt{Entity: 5, Class: "weapon_machinegun", Started: 10, Target: quake.Vec3{0, 0, 9.125}}, progress: 10}}
		s := quake.Snapshot{Frame: 11, Health: 100, InventoryKnown: true, Pickups: []quake.Object{{ID: 5, Class: "weapon_machinegun"}}}
		if meet {
			mate := quake.Vec3{100, 0, 0}
			s.Teammate = &mate
			p.World.Goal = "follow_teammate"
		} else {
			s.Inventory = []quake.InventoryItem{{Name: "Machinegun", Count: 1}}
		}
		if _, ok := p.pickupGoal(s); ok || p.pickup != nil || p.respawnRegroup != parent {
			t.Fatal("pickup retained control or replaced return parent")
		}
		want := "confirmed"
		if meet {
			want = "interrupted"
		}
		if p.World.Pickup.State != want {
			t.Fatal(p.World.Pickup)
		}
	}
}

func TestReturnPickupDetourBudget(t *testing.T) {
	from, target := quake.Vec3{}, quake.Vec3{500, 0, 0}
	if !returnPickupWithinBudget(from, quake.Vec3{80, 0, 0}, target, 80) ||
		returnPickupWithinBudget(from, quake.Vec3{100, 0, 0}, target, 100) ||
		returnPickupWithinBudget(from, quake.Vec3{-30, 0, 0}, target, 30) {
		t.Fatal("return detour budget")
	}
}

func TestBase2PickupWithoutVisiblePlayer(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires local BSP/AAS")
	}
	g, err := quake.LoadMap(root, "base2")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(root + "/maps/base2.aas")
	if err != nil {
		t.Fatal(err)
	}
	for _, returning := range []bool{false, true} {
		p := &Planner{Nav: n, World: World{Geometry: &g, Goal: "wait_for_teammate"}}
		if returning {
			p.World.Goal = "regroup_after_respawn"
			p.respawnRegroup = &respawnRegroup{target: quake.Vec3{194, 1940, -167.875}}
		}
		s := quake.Snapshot{Frame: 100, Health: 100, InventoryKnown: true, OnGround: true, Self: quake.Vec3{828, 2232, -231.875}, Pickups: []quake.Object{{ID: 5, Class: "weapon_machinegun", Origin: quake.Vec3{804, 2232, -241}}}}
		if at, ok := p.pickupGoal(s); !ok || at != healthStand(s.Pickups[0].Origin) {
			cost, valid := p.resourceRoute(s.Self, healthStand(s.Pickups[0].Origin))
			t.Fatalf("returning=%v cost=%v valid=%v pickup=%+v", returning, cost, valid, p.World.Pickup)
		}
	}
}

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
