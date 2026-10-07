package bot

func multiWeaponFixture(fixture string) bool {
	return fixture == "parasite_weapons" || fixture == "parasite_weapons-scarce"
}

func multiWeaponStock(fixture string) (bullets, shells int) {
	if fixture == "parasite_weapons-scarce" {
		return 10, 6
	}
	return 40, 20
}

func shotgunWeapon(weapon string) bool {
	return weapon == "Shotgun" || weapon == "models/weapons/v_shotg/tris.md2"
}
