package bot

import (
	"context"
	"math"
	"testing"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

func TestDiagnosticNominalAimPreservesDecisionsAndWorldMovement(t *testing.T) {
	clear, solid, track := true, uint16(2|3<<5|8<<10), 7
	o := policy.Observation{Weapon: "models/weapons/v_machn/tris.md2", Enemies: []policy.Enemy{{ID: 3, Track: &track, Relative: quake.Vec3{0, 200, 0}, Solid: &solid, ClearShot: &clear}}}
	a := policy.Action{TargetEntity: 3, TargetTrack: 7}
	delta := [3]int16{1234, 5678, 0}
	cmd := quake.UserCmd{Yaw: -5678, Forward: 300, Side: 100, Up: 200, Buttons: 1}
	got, changes := diagnosticNominalCombatAim(o, a, cmd, delta)
	if got.Buttons != cmd.Buttons || got.Up != cmd.Up || len(changes) != 3 {
		t.Fatalf("decisions/interventions changed incorrectly: %+v %+v", got, changes)
	}
	world := func(c quake.UserCmd) [2]float64 {
		y := float64(int16(uint16(c.Yaw)+uint16(delta[1]))) * 2 * math.Pi / 65536
		return [2]float64{float64(c.Forward)*math.Cos(y) + float64(c.Side)*math.Sin(y), float64(c.Forward)*math.Sin(y) - float64(c.Side)*math.Cos(y)}
	}
	before, after := world(cmd), world(got)
	if math.Hypot(before[0]-after[0], before[1]-after[1]) > 1 {
		t.Fatalf("world movement changed: %v -> %v", before, after)
	}
	if int16(uint16(got.Yaw)+uint16(delta[1])) != 16384 {
		t.Fatal("did not face the selected target using protocol delta angles")
	}
	// Never add fire even when the observed target is available.
	cmd.Buttons = 0
	got, _ = diagnosticNominalCombatAim(o, a, cmd, delta)
	if got.Buttons != 0 {
		t.Fatal("diagnostic added fire")
	}
}

func TestDiagnosticNominalAimRequiresSelectedVisibleTrack(t *testing.T) {
	clear, solid, track := true, uint16(2|3<<5|8<<10), 7
	base := policy.Observation{Weapon: "Machinegun", Enemies: []policy.Enemy{{ID: 3, Track: &track, Relative: quake.Vec3{0, 200, 0}, Solid: &solid, ClearShot: &clear}}}
	for _, name := range []string{"no-target", "wrong-track", "blocked", "unknown-box", "projectile"} {
		t.Run(name, func(t *testing.T) {
			o := base
			o.Enemies = append([]policy.Enemy(nil), base.Enemies...)
			a := policy.Action{TargetEntity: 3, TargetTrack: 7}
			switch name {
			case "no-target":
				a.TargetEntity = 0
			case "wrong-track":
				a.TargetTrack = 8
			case "blocked":
				v := false
				o.Enemies[0].ClearShot = &v
			case "unknown-box":
				o.Enemies[0].Solid = nil
			case "projectile":
				o.Weapon = "Blaster"
			}
			cmd := quake.UserCmd{Yaw: 123, Pitch: 456, Forward: 123, Buttons: 1}
			got, changes := diagnosticNominalCombatAim(o, a, cmd, [3]int16{})
			if got != cmd || len(changes) != 0 {
				t.Fatal("unavailable target changed command")
			}
		})
	}
}

func TestDiagnosticNominalAimCannotRunInOrdinaryCoop(t *testing.T) {
	err := Run(context.Background(), Config{CombatDiagnosticNominalAim: true, CombatMode: "learned"})
	if err == nil {
		t.Fatal("accepted diagnostic outside synchronous fixture")
	}
}

func TestDiagnosticNominalAimObservedWeaponNames(t *testing.T) {
	clear, solid := true, uint16(2|3<<5|8<<10)
	for _, name := range []string{"Machinegun", "Shotgun", "Super Shotgun", "models/weapons/v_machn/tris.md2", "models/weapons/v_shotg/tris.md2", "models/weapons/v_shotg2/tris.md2"} {
		o := policy.Observation{Weapon: name, Enemies: []policy.Enemy{{ID: 3, Relative: quake.Vec3{0, 200, 0}, Solid: &solid, ClearShot: &clear}}}
		_, changes := diagnosticNominalCombatAim(o, policy.Action{TargetEntity: 3}, quake.UserCmd{}, [3]int16{})
		if len(changes) == 0 {
			t.Fatalf("observed weapon did not activate diagnostic: %s", name)
		}
	}
}
