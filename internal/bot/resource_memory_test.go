package bot

import (
	"os"
	"path/filepath"
	"testing"

	"q2coopbot/internal/quake"
)

func TestResourceObservationLifecycle(t *testing.T) {
	p := &Planner{World: World{Map: "base1"}}
	item := quake.Object{ID: 7, Class: "item_health", Origin: quake.Vec3{200, 0, 15}, HealthAmount: 25}
	p.observeResources(quake.Snapshot{Frame: 10, Pickups: []quake.Object{item}})
	p.observeResources(quake.Snapshot{Frame: 11})
	r := p.resources[7]
	if r.State != "unknown" || r.LastSeen != 10 {
		t.Fatalf("lost visibility: %+v", r)
	}
	p.markResourceVisit(healthStand(item.Origin))
	p.observeResources(quake.Snapshot{Frame: 12})
	if !r.Attempted {
		t.Fatal("failed visit can be repeated without evidence")
	}
	p.observeResources(quake.Snapshot{Frame: 13, Pickups: []quake.Object{item}})
	if p.resources[7].Attempted || p.resources[7].State != "observed" {
		t.Fatal("new sighting did not refresh availability")
	}
	p.observeResources(quake.Snapshot{Frame: 614})
	if len(p.resources) != 0 {
		t.Fatal("stale memory retained")
	}
	p.observeResources(quake.Snapshot{Frame: 615, Pickups: []quake.Object{item}})
	p.setMap("", "") // Client uses this on reconnect and same-map serverdata.
	if len(p.resources) != 0 {
		t.Fatal("memory leaked into another generation")
	}
}

func TestResourceMemoryBoundAndEntityReuse(t *testing.T) {
	p := &Planner{}
	s := quake.Snapshot{Frame: 10}
	for i := 0; i < 150; i++ {
		s.Pickups = append(s.Pickups, quake.Object{ID: i, Class: "item_health"})
	}
	p.observeResources(s)
	if len(p.resources) != 128 {
		t.Fatal("memory unbounded")
	}
	p.observeResources(quake.Snapshot{Frame: 11, Pickups: []quake.Object{{ID: 149, Class: "ammo_shells"}}})
	if p.resources[149] == nil || p.resources[149].Item.Class != "ammo_shells" || p.resources[149].LastSeen != 11 {
		t.Fatal("reused entity retains old resource identity")
	}
}

func TestRememberedWeaponAndAmmoReturn(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base1 BSP/AAS")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(filepath.Join(root, "maps/base1.aas"))
	if err != nil {
		t.Fatal(err)
	}
	for _, class := range []string{"weapon_shotgun", "ammo_shells"} {
		s := quake.Snapshot{Map: "base1", Frame: 40, Health: 100, OnGround: true, InventoryKnown: true, Self: quake.Vec3{-300, -544, -79.875}}
		if class == "ammo_shells" {
			s.Inventory = []quake.InventoryItem{{Name: "Shotgun", Count: 1}, {Name: "Shells", Count: 2}}
		}
		p := &Planner{Campaign: true, Nav: n, World: World{Map: "base1", Geometry: &g, Goal: "reach_level_exit"}}
		item := quake.Object{ID: 42, Class: class, Origin: quake.Vec3{-384, -544, -88.875}}
		s.Pickups = []quake.Object{item}
		p.observeResources(s)
		s.Frame++
		s.Pickups = nil
		p.observeResources(s)
		if _, ok := p.pickupGoal(s); !ok || !p.pickup.attempt.FromMemory {
			t.Fatal("forgot useful supply", class, p.World.Pickup)
		}
		s.Frame++
		s.Self = healthStand(item.Origin)
		if class == "weapon_shotgun" {
			s.Inventory = []quake.InventoryItem{{Name: "Shotgun", Count: 1}}
		} else {
			s.Inventory[1].Count = 12
		}
		if _, ok := p.pickupGoal(s); ok || p.World.Pickup.State != "confirmed" {
			t.Fatal("inventory delta did not confirm memory pickup", p.World.Pickup)
		}
	}
}

func TestRememberedAmmoReserveAndOwnership(t *testing.T) {
	s := quake.Snapshot{InventoryKnown: true, Inventory: []quake.InventoryItem{{Name: "Shells", Count: 2}}}
	if usefulRememberedPickup(s, "ammo_shells") {
		t.Fatal("ammo without its weapon")
	}
	s.Inventory = append(s.Inventory, quake.InventoryItem{Name: "Shotgun", Count: 1})
	if !usefulRememberedPickup(s, "ammo_shells") {
		t.Fatal("low reserve ignored")
	}
	s.Inventory[0].Count = 10
	if usefulRememberedPickup(s, "ammo_shells") {
		t.Fatal("unnecessary memory detour")
	}
	s.InventoryAgeFrames = 21
	if usefulRememberedPickup(s, "weapon_machinegun") {
		t.Fatal("stale ownership used")
	}
}

func TestResourceMemoryBase1ReturnAndMissing(t *testing.T) {
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
	mate := quake.Vec3{-300, -600, -79.875}
	s := quake.Snapshot{Frame: 40, Health: 100, OnGround: true, InventoryKnown: true, Self: quake.Vec3{-300, -544, -79.875}, Teammate: &mate}
	item := quake.Object{ID: 42, Class: "item_armor_jacket", Origin: quake.Vec3{-384, -544, -88.875}}
	p := &Planner{Nav: n, World: World{Map: "base1", Geometry: &g, Goal: "follow_teammate"}}
	s.Pickups = []quake.Object{item}
	p.observeResources(s)
	s.Frame++
	s.Pickups = nil
	p.observeResources(s)
	if _, ok := p.pickupGoal(s); !ok || !p.pickup.attempt.FromMemory {
		t.Fatal("safe return to remembered armor not selected")
	}
	s.Frame += 6
	p.observeResources(s)
	if _, ok := p.pickupGoal(s); !ok {
		t.Fatal("distant absence cancels memory approach")
	}
	s.Self = healthStand(item.Origin)
	s.Frame++
	p.observeResources(s)
	s.Frame += 5
	p.observeResources(s)
	p.pickupGoal(s)
	if p.resources[42].State != "unavailable" || p.pickup != nil || p.World.Pickup.State != "unconfirmed" {
		t.Fatal("arrival without armor must not count as pickup")
	}
	s.Frame += 160
	p.observeResources(s)
	if len(p.rememberedCandidates(s)) != 0 {
		t.Fatal("absent armor retried after cooldown")
	}
	// A new observed kit is remembered even when health is full. Later need
	// selects its old location, but only with the same guarded route.
	item.Class = "item_health"
	item.HealthAmount = 25
	s.Self = quake.Vec3{-300, -544, -79.875}
	s.Pickups = []quake.Object{item}
	p.observeResources(s)
	if _, ok := p.healthGoal(s); ok {
		t.Fatal("full health spends remembered kit")
	}
	s.Frame++
	s.Health = 40
	s.Pickups = nil
	p.observeResources(s)
	if at, ok := p.healthGoal(s); !ok || at != healthStand(item.Origin) {
		t.Fatal("needed kit not selected from memory")
	}
	p.Campaign = true
	s.Teammate = nil
	p.resources[42].Attempted = false
	if at, ok := p.healthGoal(s); !ok || at != healthStand(item.Origin) {
		t.Fatal("solo campaign cannot return to remembered health")
	}
	s.Health = 100
	if _, ok := p.healthGoal(s); ok {
		t.Fatal("campaign spends kit without need")
	}
	s.Health = 40
	p.Nav = nil
	p.resources[42].Attempted = false
	if len(p.rememberedCandidates(s)) != 0 {
		t.Fatal("memory bypasses navigation evidence")
	}
}

func TestBase1HealthRejectsExpensiveUpperFloor(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires local base1 assets")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(filepath.Join(root, "maps/base1.aas"))
	if err != nil {
		t.Fatal(err)
	}
	p := &Planner{Nav: n, World: World{Geometry: &g}}
	upper := quake.Object{Class: "item_health", Origin: quake.Vec3{1128, -96, -40.875}}
	lower := quake.Object{Class: "item_health", Origin: quake.Vec3{1056, 344, -176.875}}
	s := quake.Snapshot{Frame: 100, Health: 40, Self: quake.Vec3{940, 75, -167.875}, Pickups: []quake.Object{upper, lower}}
	at, ok := p.healthGoal(s)
	if !ok || at != healthStand(lower.Origin) {
		for _, item := range s.Pickups {
			cost, valid := p.resourceRoute(s.Self, healthStand(item.Origin))
			t.Logf("item=%v cost=%v valid=%v", item.Origin, cost, valid)
		}
		t.Fatalf("goal=%v ok=%v", at, ok)
	}
}

func TestBase1MemoryBeyondObservationPreservesRouteBudget(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires local base1 assets")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(filepath.Join(root, "maps/base1.aas"))
	if err != nil {
		t.Fatal(err)
	}
	mate := quake.Vec3{960, 248, -167.875}
	item := quake.Object{ID: 180, Class: "item_health", Origin: quake.Vec3{1056, 344, -176.875}}
	p := &Planner{Nav: n, World: World{Geometry: &g}, resources: map[int]*ResourceMemory{180: {Item: item, State: "unknown", LastSeen: 10}}}
	s := quake.Snapshot{Frame: 20, Health: 40, OnGround: true, Self: quake.Vec3{960, -40, -167.875}, Teammate: &mate}
	if quake.Distance(s.Self, item.Origin) <= 384 {
		t.Fatal("fixture is still within observation radius")
	}
	// This location is within the expanded radius, but its walking route
	// exceeds the route budget. Euclidean proximity alone must not admit it.
	if len(p.rememberedCandidates(s)) != 0 {
		t.Fatal("expanded radius bypassed the walking route budget")
	}
	s.Self[1] = -200
	if len(p.rememberedCandidates(s)) != 0 {
		t.Fatal("distant return accepted")
	}
}
