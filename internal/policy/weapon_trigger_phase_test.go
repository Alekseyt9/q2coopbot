package policy

import "testing"

func TestObservedTriggerPhaseSeparatesWindupRecoveryAndAmmo(t *testing.T) {
	for _, test := range []struct {
		weapon      string
		frame, ammo int
		ready       bool
	}{
		{"Grenades", 3, 5, false}, {"Grenades", 11, 1, true}, {"Grenades", 11, 0, false},
		{"models/weapons/v_machn/tris.md2", 4, 1, true}, {"Machinegun", 6, 10, false}, {"Machinegun", 7, 0, false},
		{"Shotgun", 19, 1, false}, {"Shotgun", 20, 1, true}, {"Super Shotgun", 19, 1, false}, {"Super Shotgun", 19, 2, true},
		{"Railgun", 19, 1, false}, {"Railgun", 20, 1, true}, {"BFG10K", 34, 49, false}, {"BFG10K", 34, 50, true},
		{"HyperBlaster", 6, 1, true}, {"Chaingun", 14, 1, true}, {"Chaingun", 8, 10, false},
		{"Blaster", 10, 0, true}, {"Unknown", 10, 100, false},
	} {
		if got := ObservedWeaponTriggerPhase(test.weapon, test.frame, test.ammo); got != test.ready {
			t.Fatalf("%s frame%d ammo%d: %v", test.weapon, test.frame, test.ammo, got)
		}
	}
}
