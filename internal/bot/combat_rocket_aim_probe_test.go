package bot

import (
	"path/filepath"
	"testing"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"q2coopbot/internal/rocketpractice"
)

func TestRocketAimProbeRestrictions(t *testing.T) {
	base := Config{Host: "127.0.0.1", CombatMode: "learned", CombatCapture: true, TestSynchronous: true, TestCombatOnly: true, CombatCheckedWallSlide: true, TestRocketAimProbeFrame: 99, TestRocketAimProbePoint: "1,2,3"}
	if p, err := validateRocketAimProbe(base, policy.StockWeaponsFireCombatExecution); err != nil || p == nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*Config){
		func(c *Config) { c.Host = "192.0.2.1" }, func(c *Config) { c.TestSynchronous = false }, func(c *Config) { c.TestCombatOnly = false },
		func(c *Config) { c.CombatCapture = false }, func(c *Config) { c.TestRocketAimProbeFrame = 0 }, func(c *Config) { c.TestRocketAimProbeFrame = 1000001 },
		func(c *Config) { c.TestRocketAimProbePoint = "NaN,2,3" }, func(c *Config) { c.TestGrenadeReleaseProbeFrame = 99 },
		func(c *Config) { c.CombatActionQualityGuard = true }, func(c *Config) { c.TestCombatAdvice = true },
	} {
		c := base
		mutate(&c)
		if _, err := validateRocketAimProbe(c, policy.StockWeaponsFireCombatExecution); err == nil {
			t.Fatal("invalid diagnostic config admitted", c)
		}
	}
}

func TestRocketAimProbeChangesOnlyAnglesAtReadyStationaryFrame(t *testing.T) {
	root, _ := filepath.Abs(filepath.Join("..", ".."))
	world, err := quake.LoadMap(filepath.Join(root, "workspace/runtime/q2go/baseq2"), "base1")
	if err != nil {
		t.Skipf("local BSP needed: %v", err)
	}
	clear := true
	solid := uint16(8290)
	o := policy.Observation{Version: policy.ObservationVersion, Identity: policy.Identity{Map: "base1", Frame: 99}, Health: 100, Position: quake.Vec3{-1632, 1288, 120.125}, Enemies: []policy.Enemy{{ID: 2, Relative: quake.Vec3{-192, 0, 0}, Distance: 192, ClearShot: &clear, Solid: &solid}}}
	options, err := rocketpractice.Generate(o, &world)
	if err != nil {
		t.Fatal(err)
	}
	var point quake.Vec3
	found := false
	for _, c := range options {
		if c.Kind == "floor" {
			point = c.Point
			found = true
			break
		}
	}
	if !found {
		t.Fatal("no floor candidate")
	}
	s := quake.Snapshot{Self: o.Position, Health: 100, Ammo: 5, GunFrame: 14, Weapon: "Rocket Launcher", DeltaAngles: [3]int16{123, 456, 0}}
	cmd := quake.UserCmd{Pitch: 7, Yaw: 12, Up: 400, Buttons: 3, Msec: 100, Light: 62}
	got, changes := diagnosticRocketSurfaceAim(o, s, &world, cmd, point)
	want := cmd
	want.Pitch = got.Pitch
	want.Yaw = got.Yaw
	if got != want || len(changes) == 0 {
		t.Fatal("probe changed non-aim fields or did not apply", got, changes)
	}
	for _, mutate := range []func(*quake.Snapshot, *quake.UserCmd){
		func(s *quake.Snapshot, c *quake.UserCmd) { c.Forward = 1 }, func(s *quake.Snapshot, c *quake.UserCmd) { c.Side = 1 },
		func(s *quake.Snapshot, c *quake.UserCmd) { c.Buttons = 0 }, func(s *quake.Snapshot, c *quake.UserCmd) { s.GunFrame = 6 },
		func(s *quake.Snapshot, c *quake.UserCmd) { s.Weapon = "Grenade Launcher" },
	} {
		state, before := s, cmd
		mutate(&state, &before)
		after, changes := diagnosticRocketSurfaceAim(o, state, &world, before, point)
		if after != before || len(changes) != 0 {
			t.Fatal("unqualified probe applied")
		}
	}
	if got, changes := diagnosticRocketSurfaceAim(o, s, &world, cmd, quake.Vec3{1, 2, 3}); got != cmd || len(changes) != 0 {
		t.Fatal("point outside observed candidate set admitted")
	}
}
