package bot

import (
	"context"
	"strings"
	"testing"

	"q2coopbot/internal/quake"
)

func TestShotgunTeacherWaitsForFreshLoadedWeapon(t *testing.T) {
	s := quake.Snapshot{Weapon: "models/weapons/v_shotg/tris.md2", Ammo: 20, InventoryKnown: true, InventoryAgeFrames: 1, Inventory: []quake.InventoryItem{{Name: "Shells", Count: 20}}}
	if !synchronousFixtureWeaponReady("parasite_shotgun", s) {
		t.Fatal("loaded Shotgun not ready")
	}
	for _, name := range []string{"wrong_weapon", "wrong_ammo", "stale_inventory", "missing_shells", "unknown_fixture"} {
		t.Run(name, func(t *testing.T) {
			bad := s
			fixture := "parasite_shotgun"
			switch name {
			case "wrong_weapon":
				bad.Weapon = "Blaster"
			case "wrong_ammo":
				bad.Ammo = 19
			case "stale_inventory":
				bad.InventoryAgeFrames = 3
			case "missing_shells":
				bad.Inventory = nil
			case "unknown_fixture":
				fixture = "other"
			}
			if synchronousFixtureWeaponReady(fixture, bad) {
				t.Fatal("invalid equipment released")
			}
		})
	}
}

func TestShotgunFixtureRejectsUnsynchronizedUse(t *testing.T) {
	for _, mode := range []string{"rules", "learned", "learned-shadow"} {
		cfg := Config{Host: "127.0.0.1", CombatMode: mode, TestWeaponSwitchFixture: "parasite_shotgun"}
		err := Run(context.Background(), cfg)
		if err == nil || mode == "rules" && !strings.Contains(err.Error(), "fixed Shotgun fixture") || mode != "rules" && !strings.Contains(err.Error(), "direct/shadow") {
			t.Fatalf("unexpected validation %s: %v", mode, err)
		}
	}
}

func TestShotgunFixtureAdmitsIsolatedLearnedCapture(t *testing.T) {
	// A nonexistent provider must fail at loading, after the isolation gate,
	// without a socket or any numerical policy evaluation.
	cfg := Config{Host: "127.0.0.1", CombatMode: "learned", TestWeaponSwitchFixture: "parasite_shotgun",
		TestSynchronous: true, TestCombatBarrier: true, FramePaced: true, CombatCapture: true,
		TestTeleport: "32,-224,24", CombatProviderFile: "nonexistent-ready-shotgun-provider.json"}
	if err := Run(context.Background(), cfg); err == nil || !strings.Contains(err.Error(), "combat provider:") {
		t.Fatalf("isolated learned Shotgun rejected before provider loading: %v", err)
	}
	cfg.TestSynchronous = false
	if err := Run(context.Background(), cfg); err == nil || !strings.Contains(err.Error(), "direct/shadow") {
		t.Fatalf("unsynchronized learned Shotgun admitted: %v", err)
	}
}

func TestShotgunGeneratedRulesChoiceWaitsForReleaseAndPreservesTeacher(t *testing.T) {
	c := &Client{testWeaponSwitchFixture: "parasite_shotgun", testSynchronous: true, testCombatGo: true, planner: &Planner{}}
	c.planner.testCombatOnly = true
	c.combatControl.mode = "rules"
	if !c.shotgunFixtureAllowsRulesWeaponChoice() {
		t.Fatal("released generated rules capture cannot use ordinary weapon selection")
	}
	for _, change := range []func(*Client){
		func(c *Client) { c.testCombatGo = false },
		func(c *Client) { c.testSynchronous = false },
		func(c *Client) { c.planner.testCombatOnly = false },
		func(c *Client) { c.testTeacherVertical = true },
		func(c *Client) { c.combatControl.mode = "learned" },
	} {
		other := *c
		planner := *c.planner
		other.planner = &planner
		change(&other)
		if other.shotgunFixtureAllowsRulesWeaponChoice() {
			t.Fatal("fixed teacher/setup or learned fallback acquired rules weapon choice")
		}
	}
}
