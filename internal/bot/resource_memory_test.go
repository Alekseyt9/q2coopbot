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
	if p.resources[149] != nil {
		t.Fatal("reused entity retains old resource")
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
	p.Nav = nil
	if len(p.rememberedCandidates(s)) != 0 {
		t.Fatal("memory bypasses navigation evidence")
	}
}
