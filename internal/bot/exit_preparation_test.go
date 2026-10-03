package bot

import (
	"os"
	"path/filepath"
	"testing"

	"q2coopbot/internal/quake"
)

func TestExitPreparationReservesAndSharedClock(t *testing.T) {
	p := &Planner{Campaign: true, World: World{Campaign: &CampaignDecision{Exit: &quake.MapExit{Center: quake.Vec3{}}}}}
	s := quake.Snapshot{Map: "base1", Frame: 100, Health: 60, Armor: 0, OnGround: true, InventoryKnown: true, Inventory: []quake.InventoryItem{{Name: "Shotgun", Count: 1}, {Name: "Shells", Count: 1}}}
	p.updateExitPreparation(s)
	d := p.exitPreparation
	if d.State != "collecting" || d.MissingHealth != 15 || d.MissingArmor != 25 || d.MissingAmmo["Shells"] != 9 || d.MissingWeapon {
		t.Fatalf("wrong combined needs: %+v", d)
	}
	s.Frame += 20
	s.Health, s.Armor = 85, 50
	s.Inventory[1].Count = 11
	p.updateExitPreparation(s)
	if d.State != "ready" || d.SpentFrames != 20 {
		t.Fatalf("not ready: %+v", d)
	}
	// Fresh damage and ammunition use may resume preparation, within the
	// same clock; leaving and returning cannot replenish the budget.
	s.Frame += 10
	s.Health = 60
	s.Inventory[1].Count = 1
	s.Self[0] = 1000
	p.updateExitPreparation(s)
	s.Frame += 170
	s.Self[0] = 0
	p.updateExitPreparation(s)
	if d.State != "budget_exhausted" || d.SpentFrames != 200 || p.preparingForExit(s) || p.preparingSuppliesForExit(s) {
		t.Fatalf("budget restarted: %+v", d)
	}
	item := quake.Object{Class: "ammo_shells"}
	if p.exitPickupAllowed(s, item, quake.Vec3{200, 0, 0}, 200, false) || p.exitPickupAllowed(s, item, s.Self, 0, true) {
		t.Fatal("exhausted preparation started a new detour")
	}
	if !p.exitPickupAllowed(s, item, quake.Vec3{20, 0, 0}, 20, false) {
		t.Fatal("blocked free visible pickup")
	}
	s.InventoryAgeFrames = 21
	d.SpentFrames = 20
	p.updateExitPreparation(s)
	if d.State == "ready" || d.InventoryKnown {
		t.Fatal("stale inventory claimed readiness")
	}
	// Setup observation must not start the shared clock.
	p.exitPreparation = nil
	p.testSetupHold = true
	p.updateExitPreparation(s)
	if p.exitPreparation != nil {
		t.Fatal("setup consumed preparation budget")
	}
}

func TestBase1PreparationDeadlinePreservesEmergencyAndExit(t *testing.T) {
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
	s := quake.Snapshot{Map: "base1", Frame: 300, Health: 60, OnGround: true, Self: quake.Vec3{-1488, 1800, -23.875}, Pickups: []quake.Object{{ID: 42, Class: "item_health", Origin: quake.Vec3{-1500, 1700, -32.875}, HealthAmount: 25}}}
	p := &Planner{Campaign: true, CampaignNextMap: "base2", Nav: n, World: World{Geometry: &g}, exitPreparation: &ExitPreparation{Map: "base1", SpentFrames: 200, lastFrame: 300}}
	if _, ok := p.campaignGoal(s); !ok {
		t.Fatal("exit unavailable")
	}
	if _, ok := p.healthGoal(s); ok {
		t.Fatal("optional healing ignores shared deadline")
	}
	s.Health = 40
	if _, ok := p.healthGoal(s); !ok {
		t.Fatal("deadline suppresses emergency health")
	}
	s.Health = 60
	p.World.Goal = "recover_health"
	p.healthActive = true
	p.healthTarget = healthStand(s.Pickups[0].Origin)
	p.healthAt = 100
	p.healthStarted = 100
	goal := p.budgetHealthGoal(s, p.healthTarget)
	if p.World.Goal != "reach_level_exit" || goal == p.healthTarget {
		t.Fatal("failed health visit retained the health goal instead of resuming exit")
	}
}

func TestBase1PreparationDeadlineCancelsActivePickup(t *testing.T) {
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
	s := quake.Snapshot{Map: "base1", Frame: 100, Health: 85, Armor: 50, OnGround: true, InventoryKnown: true, Self: quake.Vec3{-1488, 1800, -23.875}, Inventory: []quake.InventoryItem{{Name: "Shotgun", Count: 1}, {Name: "Shells", Count: 1}}, Pickups: []quake.Object{{ID: 42, Class: "ammo_shells", Origin: quake.Vec3{-1450, 1760, -32.875}}}}
	p := &Planner{Campaign: true, CampaignNextMap: "base2", Nav: n, World: World{Geometry: &g, Goal: "reach_level_exit"}}
	p.campaignGoal(s)
	p.observeResources(s)
	if _, ok := p.pickupGoal(s); !ok {
		t.Fatal("needed shells not acquired")
	}
	s.Frame += 200
	p.campaignGoal(s)
	if _, ok := p.pickupGoal(s); ok || p.pickup != nil || p.World.Pickup.State != "preparation_complete" {
		t.Fatal("deadline retained active resource detour")
	}
	s.Frame += 10
	p.campaignGoal(s)
	if _, ok := p.pickupGoal(s); ok {
		t.Fatal("exhausted preparation retried pickup")
	}
}

func TestExitPreparationCheckpointDoesNotRefillBudget(t *testing.T) {
	p := &Planner{Campaign: true, CampaignNextMap: "base2", campaignMap: "base1", campaignDestination: "base2", exitPreparation: &ExitPreparation{Map: "base1", SpentFrames: 190}, World: World{Snapshot: quake.Snapshot{Map: "base1", Frame: 100, Health: 60, OnGround: true}}}
	state, err := p.CaptureCheckpoint()
	if err != nil {
		t.Fatal(err)
	}
	fresh := quake.Snapshot{Map: "base1", Frame: 5, Health: 60, OnGround: true}
	q := &Planner{Campaign: true, CampaignNextMap: "base2"}
	if err = q.RestoreCheckpoint(state, fresh, ""); err != nil {
		t.Fatal(err)
	}
	q.World.Campaign = &CampaignDecision{Exit: &quake.MapExit{}}
	fresh.Frame += 10
	q.updateExitPreparation(fresh)
	if q.exitPreparation.SpentFrames != 200 || q.exitPreparation.State != "budget_exhausted" {
		t.Fatal("restore refilled clock")
	}
	for _, bad := range []int{-1, 201} {
		state.Campaign.PreparationSpent = &bad
		if state.validate("base1") == nil {
			t.Fatal("invalid budget accepted")
		}
	}
}
