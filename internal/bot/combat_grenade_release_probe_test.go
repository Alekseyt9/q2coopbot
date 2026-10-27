package bot

import (
	"context"
	"strings"
	"testing"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

func TestGrenadeReleaseProbeNativeBitOnly(t *testing.T) {
	cmd := quake.UserCmd{Forward: 123, Side: -75, Up: -400, Buttons: 3}
	cmd.Pitch, cmd.Yaw, cmd.Roll = 321, 1234, 19
	cmd.Impulse, cmd.Msec, cmd.Light = 4, 100, 80
	s := quake.Snapshot{Health: 80, Weapon: "models/weapons/v_handgr/tris.md2", GunFrame: 11}
	want := cmd
	want.Buttons &^= 1
	got, changes := diagnosticGrenadeRelease(s, cmd)
	if got != want || len(changes) != 1 || changes[0].Reason != "diagnostic_grenade_release" {
		t.Fatal("release changed a non-attack field", got, want, changes)
	}
	for _, mutate := range []func(*quake.Snapshot, *quake.UserCmd){
		func(s *quake.Snapshot, c *quake.UserCmd) { s.Health = 0 },
		func(s *quake.Snapshot, c *quake.UserCmd) { s.GunFrame = 10 },
		func(s *quake.Snapshot, c *quake.UserCmd) { s.GunFrame = 12 },
		func(s *quake.Snapshot, c *quake.UserCmd) { s.Weapon = "Grenade Launcher" },
		func(s *quake.Snapshot, c *quake.UserCmd) { c.Buttons &^= 1 },
	} {
		state, before := s, cmd
		mutate(&state, &before)
		after, changes := diagnosticGrenadeRelease(state, before)
		if after != before || len(changes) != 0 {
			t.Fatal("probe modified an unqualified command", state, before, after)
		}
	}
}

func TestGrenadeReleaseProbeRejectsOrdinarySessions(t *testing.T) {
	base := Config{Host: "127.0.0.1", CombatMode: "learned", CombatCapture: true, TestSynchronous: true,
		TestCombatOnly: true, CombatCheckedWallSlide: true, CombatExecutionContract: policy.StockWeaponsFireCombatExecution,
		TestGrenadeReleaseProbeFrame: 132}
	for _, mutate := range []func(*Config){
		func(c *Config) { c.Host = "192.0.2.1" },
		func(c *Config) { c.TestSynchronous = false },
		func(c *Config) { c.TestCombatOnly = false },
		func(c *Config) { c.CombatCapture = false },
		func(c *Config) { c.TestGrenadeReleaseProbeFrame = -1 },
		func(c *Config) { c.TestGrenadeReleaseProbeFrame = 1000001 },
		func(c *Config) { c.TestForcedDuckProbeFrame = 132 },
		func(c *Config) { c.CombatExecutionContract = policy.LegacyCombatExecution },
	} {
		cfg := base
		mutate(&cfg)
		err := Run(context.Background(), cfg)
		if err == nil || !(strings.Contains(err.Error(), "grenade release probe") || strings.Contains(err.Error(), "checked combat execution contract")) {
			t.Fatal("invalid probe reached runtime", err)
		}
	}
}
