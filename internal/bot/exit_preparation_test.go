package bot

import (
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
