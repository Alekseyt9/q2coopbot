package learningenv

import (
	"math"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"testing"
)

func TestResetVerifiesObservedFieldsAndKeepsHiddenStateUnconfirmed(t *testing.T) {
	s := damageStep(10)
	o := s.Observation
	o.Version = policy.ObservationVersion
	o.Health = 100
	o.Ammo = 20
	o.OnGround = true
	o.Weapon = "models/weapons/v_shotg/tris.md2"
	o.Position = quake.Vec3{32, -224, 24.125}
	visible := true
	o.Enemies = []policy.Enemy{{Class: "monster_parasite", Relative: quake.Vec3{168, 0, -.125}, ClearShot: &visible}}
	expected := ResetExpectation{Version: ResetVersion, Map: "base1", Position: quake.Vec3{32, -224, 24}, Health: 100, Ammo: 20, Weapon: "Shotgun", EnemyClass: "monster_parasite", EnemyPosition: quake.Vec3{200, -224, 24}}
	r := VerifyReset(o, expected)
	if !r.ObservedFieldsConfirmed || r.FullServerResetConfirmed || len(r.Unverified) == 0 || r.Error() != nil {
		t.Fatal(r)
	}
	for _, change := range []func(*policy.Observation){
		func(o *policy.Observation) { o.Identity.Life++ }, func(o *policy.Observation) { o.Identity.Map = "base2" },
		func(o *policy.Observation) { o.Health-- }, func(o *policy.Observation) { o.Armor++ }, func(o *policy.Observation) { o.Ammo-- },
		func(o *policy.Observation) { o.OnGround = false }, func(o *policy.Observation) { o.Ducked = true },
		func(o *policy.Observation) { o.Position[0] += 2 }, func(o *policy.Observation) { o.Position[0] = math.NaN() },
		func(o *policy.Observation) { o.Weapon = "Blaster" }, func(o *policy.Observation) { o.Enemies = nil },
		func(o *policy.Observation) { o.AgeMS = 301 },
	} {
		copy := o
		change(&copy)
		if r := VerifyReset(copy, expected); r.ObservedFieldsConfirmed || r.Error() == nil {
			t.Fatal(r)
		}
	}
	expected.Position[0] = math.NaN()
	if r := VerifyReset(o, expected); r.ObservedFieldsConfirmed {
		t.Fatal("invalid expectation accepted")
	}
}

func TestMachinegunResetRejectsWrongAmmo(t *testing.T) {
	o := damageStep(10).Observation
	o.Version = policy.ObservationVersion
	o.Health, o.Ammo, o.OnGround = 100, 100, true
	o.Weapon = "models/weapons/v_machn/tris.md2"
	o.Position = quake.Vec3{32, -224, 24.125}
	visible := true
	o.Enemies = []policy.Enemy{{Class: "monster_parasite", Relative: quake.Vec3{168, 0, -.125}, ClearShot: &visible}}
	expected := ResetExpectation{Version: ResetVersion, Map: "base1", Position: quake.Vec3{32, -224, 24}, Health: 100, Ammo: 100, Weapon: "Machinegun", EnemyClass: "monster_parasite", EnemyPosition: quake.Vec3{200, -224, 24}}
	if r := VerifyReset(o, expected); !r.ObservedFieldsConfirmed || r.FullServerResetConfirmed {
		t.Fatal(r)
	}
	o.Ammo = 99
	if r := VerifyReset(o, expected); r.ObservedFieldsConfirmed || r.Reason != "reset_resources_mismatch" {
		t.Fatal("wrong bullet count accepted", r)
	}
}
