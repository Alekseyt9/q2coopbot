package policy

// ObservedWeaponTriggerPhase is a conservative stock-weapon capture predicate.
// It uses public gun-frame/ammo observations only. A pending use request can
// still preempt a ready weapon, so paired native rollouts remain authoritative.
func ObservedWeaponTriggerPhase(observed string, frame, ammo int) bool {
	name := EquippedWeaponName(observed)
	minimum := 1
	if name == "Blaster" {
		minimum = 0
	} else if name == "Super Shotgun" {
		minimum = 2
	} else if name == "BFG10K" {
		minimum = 50
	}
	if name == "" || ammo < minimum {
		return false
	}
	// Trigger-sensitive firing phases avoid waiting for an automatic weapon's
	// idle loop. Hand-grenade11 is an observed hold/release phase, not fuse age.
	switch name {
	case "Grenades":
		return frame == 11 || frame >= 16 && frame <= 48
	case "Machinegun":
		if frame == 4 || frame == 5 {
			return true
		}
	case "HyperBlaster":
		if frame >= 6 && frame <= 11 {
			return true
		}
	case "Chaingun":
		if frame == 14 || frame == 21 {
			return true
		}
	}
	// Weapon_Generic becomes ready at FRAME_IDLE_FIRST+1 after firing. Exclude
	// the ambiguous first idle frame shared with the firing/activation boundary.
	idle := map[string][2]int{
		"Blaster": {10, 52}, "Shotgun": {20, 36}, "Super Shotgun": {19, 57},
		"Machinegun": {7, 45}, "Chaingun": {33, 61}, "Grenade Launcher": {18, 59},
		"Rocket Launcher": {14, 50}, "HyperBlaster": {22, 49}, "Railgun": {20, 56}, "BFG10K": {34, 55},
	}
	range_, ok := idle[name]
	return ok && frame >= range_[0] && frame <= range_[1]
}
