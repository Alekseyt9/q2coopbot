package bot

import (
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"strings"
)

// WeaponTrainingAdvice exposes existing rules preferences to offline training.
// It reads only one client observation, does not execute commands, and is never
// called by the learned runtime. Missing preference does not mean "optimal keep".
func WeaponTrainingAdvice(o policy.Observation) (string, string, error) {
	mask, err := policy.WeaponAvailability(o)
	if err != nil {
		return "", "", err
	}
	if o.Health <= 0 || o.Inventory == nil || o.InventoryAgeFrames == nil || *o.InventoryAgeFrames < 0 || *o.InventoryAgeFrames > 20 {
		return "", "", nil
	}
	s := quake.Snapshot{Map: o.Identity.Map, Frame: o.Identity.Frame, Self: o.Position, Health: int16(o.Health), Ammo: int16(o.Ammo), Weapon: o.Weapon, GunFrame: int(o.GunFrame), InventoryKnown: true, InventoryAgeFrames: *o.InventoryAgeFrames, Inventory: append([]quake.InventoryItem(nil), (*o.Inventory)...)}
	for _, e := range o.Enemies {
		s.Enemies = append(s.Enemies, quake.Object{Class: e.Class, Origin: quake.Vec3{o.Position[0] + e.Relative[0], o.Position[1] + e.Relative[1], o.Position[2] + e.Relative[2]}, ClearShot: e.ClearShot})
	}
	name, reason := "", ""
	if isHandGrenade(s.Weapon) {
		if s.GunFrame >= 1 && s.GunFrame <= 15 {
			return "", "hand_grenade_wait_release", nil
		}
		name, reason = "Blaster", "hand_grenade_requires_safe_throw"
	} else if s.Weapon != "Blaster" && s.Weapon != "models/weapons/v_blast/tris.md2" && s.Ammo <= 0 {
		name, reason = strings.TrimPrefix(stockedWeapon(s), "use "), "empty_weapon"
	} else {
		name, reason = economyWeapon(s)
	}
	if reason == "" {
		return "", "", nil
	}
	models := []string{"v_blast", "v_shotg", "v_shotg2", "v_machn", "v_chain", "v_launch", "v_rocket", "v_hyperb", "v_rail", "v_bfg", "v_handgr"}
	for i, w := range policy.WeaponChoices()[1:] {
		if name == w && mask[i+1] {
			if s.Weapon == w || s.Weapon == "models/weapons/"+models[i]+"/tris.md2" {
				return "", reason, nil
			}
			return name, reason, nil
		}
	}
	return "", "", nil
}
