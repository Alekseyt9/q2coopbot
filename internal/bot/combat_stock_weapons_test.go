package bot

import (
	"context"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"strings"
	"testing"
)

// Fixed commands and BSP only; no neural inference or fitting.
func TestStockWeaponExecutionPreservesLegacyAndNativeCommand(t *testing.T) {
	c := policyClient(t, "learned")
	s := c.planner.World.Snapshot
	command := quake.UserCmd{Buttons: 1, Pitch: 100, Yaw: 200, Msec: 100}
	models := [...]string{"v_blast", "v_shotg", "v_shotg2", "v_machn", "v_chain", "v_launch", "v_rocket", "v_hyperb", "v_rail", "v_bfg", "v_handgr"}
	for i, name := range policy.WeaponChoices()[1:] {
		for _, weapon := range []string{name, "models/weapons/" + models[i] + "/tris.md2"} {
			s.Weapon = weapon
			legacy, _ := c.planner.guardDirectCombatMotion(s, command, true)
			wantLegacy := weapon == "Blaster" || directHitscanWeapon(weapon)
			if (legacy.Buttons&1 != 0) != wantLegacy {
				t.Fatalf("legacy changed: %s", weapon)
			}
			actual, changes := c.planner.guardDirectCombatEnvironment(s, command, true, true)
			if actual != command || len(changes) != 0 {
				t.Fatalf("stock command changed: %s: %+v %+v", weapon, actual, changes)
			}
		}
	}
	s.Weapon = "unknown_mod_weapon"
	actual, _ := c.planner.guardDirectCombatEnvironment(s, command, true, true)
	if actual.Buttons&1 != 0 {
		t.Fatal("Unknown weapon fired")
	}
	s.Weapon = "Rocket Launcher"
	s.Teammate = &quake.Vec3{100, 100, 100}
	actual, changes := c.planner.guardDirectCombatEnvironment(s, command, true, true)
	if actual.Buttons&1 != 0 || len(changes) == 0 || changes[0].Reason != "stock_weapons_solo_only" {
		t.Fatal("Solo contract admitted a teammate")
	}
	p := &Planner{}
	s.Teammate = nil
	actual, _ = p.guardDirectCombatEnvironment(s, command, true, true)
	if actual.Buttons&1 != 0 {
		t.Fatal("Missing geometry admitted fire")
	}
}

func TestStockWeaponExecutionRejectsNonlocalAndUncapturedSessions(t *testing.T) {
	base := Config{CombatExecutionContract: policy.StockWeaponsFireCombatExecution, Host: "127.0.0.1", CombatMode: "learned", CombatCapture: true, TestSynchronous: true, TestCombatOnly: true, CombatCheckedWallSlide: true}
	for _, mutate := range []func(*Config){
		func(c *Config) { c.Host = "192.0.2.1" }, func(c *Config) { c.CombatMode = "rules" },
		func(c *Config) { c.CombatCapture = false }, func(c *Config) { c.TestCombatOnly = false },
		func(c *Config) { c.TestSynchronous = false }, func(c *Config) { c.CombatCheckedWallSlide = false },
	} {
		c := base
		mutate(&c)
		if err := Run(context.Background(), c); err == nil || !strings.Contains(err.Error(), "execution contract") {
			t.Fatal("Invalid stock execution session admitted")
		}
	}
}
