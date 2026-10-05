package bot

import (
	"q2coopbot/internal/quake"
	"testing"
)

func economySnapshot() quake.Snapshot {
	visible := true
	return quake.Snapshot{Map: "base1", Frame: 10, Health: 100, Weapon: "models/weapons/v_rail/tris.md2", Ammo: 10, InventoryKnown: true, Inventory: []quake.InventoryItem{{Name: "Blaster", Count: 1}, {Name: "Shotgun", Count: 1}, {Name: "Shells", Count: 20}, {Name: "Railgun", Count: 1}, {Name: "Slugs", Count: 10}, {Name: "Machinegun", Count: 1}, {Name: "Bullets", Count: 50}, {Name: "HyperBlaster", Count: 1}, {Name: "Cells", Count: 40}}, Enemies: []quake.Object{{Class: "monster_soldier", Origin: quake.Vec3{200, 0, 0}, ClearShot: &visible}}}
}

func TestEconomyWeapon(t *testing.T) {
	for _, tc := range []struct {
		name, want string
		change     func(*quake.Snapshot)
	}{
		{"weak close", "Shotgun", func(s *quake.Snapshot) {}},
		{"weak distant", "Machinegun", func(s *quake.Snapshot) { s.Enemies[0].Origin[0] = 350 }},
		{"reserve scarce", "Blaster", func(s *quake.Snapshot) { s.Inventory[2].Count = 4; s.Inventory[6].Count = 20 }},
		{"heavy distant", "Railgun", func(s *quake.Snapshot) { s.Enemies[0].Class = "monster_tank"; s.Enemies[0].Origin[0] = 400 }},
		{"heavy close", "HyperBlaster", func(s *quake.Snapshot) { s.Enemies[0].Class = "monster_tank" }},
		{"armed close", "Shotgun", func(s *quake.Snapshot) { s.Enemies[0].Class = "monster_infantry"; s.Weapon = "Blaster" }},
		{"unknown inventory", "", func(s *quake.Snapshot) { s.InventoryKnown = false }},
		{"stale inventory", "", func(s *quake.Snapshot) { s.InventoryAgeFrames = 21 }},
		{"critical health", "", func(s *quake.Snapshot) { s.Health = 30 }},
		{"unknown enemy", "", func(s *quake.Snapshot) { s.Enemies[0].Class = "monster_custom" }},
		{"occluded", "", func(s *quake.Snapshot) { s.Enemies[0].ClearShot = nil }},
		{"multiple weak", "", func(s *quake.Snapshot) { s.Enemies = append(s.Enemies, s.Enemies[0]) }},
		{"keep cheap", "", func(s *quake.Snapshot) { s.Weapon = "Blaster" }},
		{"too distant weak", "", func(s *quake.Snapshot) { s.Enemies[0].Origin[0] = 500 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := economySnapshot()
			tc.change(&s)
			got, _ := economyWeapon(s)
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestParasiteWeaponSupportsRetreatInsteadOfShotgunApproach(t *testing.T) {
	for _, tc := range []struct {
		name, want string
		distance   float64
		change     func(*quake.Snapshot)
	}{
		{"close shotgun", "Machinegun", 120, func(s *quake.Snapshot) {}},
		{"distant shotgun", "Machinegun", 360, func(s *quake.Snapshot) {}},
		{"critical health", "Machinegun", 120, func(s *quake.Snapshot) { s.Health = 20 }},
		{"keep loaded machinegun", "", 360, func(s *quake.Snapshot) { s.Weapon = "models/weapons/v_machn/tris.md2" }},
		{"empty bullets", "HyperBlaster", 360, func(s *quake.Snapshot) { s.Inventory[6].Count = 0 }},
		{"rail at retreat range", "Railgun", 261, func(s *quake.Snapshot) { s.Inventory[6].Count = 0; s.Inventory[8].Count = 0 }},
		{"rail too close", "Blaster", 255, func(s *quake.Snapshot) { s.Inventory[6].Count = 0; s.Inventory[8].Count = 0 }},
		{"keep loaded rail at range", "", 261, func(s *quake.Snapshot) { s.Weapon = "models/weapons/v_rail/tris.md2" }},
		{"close loaded rail switch", "Machinegun", 255, func(s *quake.Snapshot) { s.Weapon = "models/weapons/v_rail/tris.md2" }},
		{"keep loaded hyper", "", 168, func(s *quake.Snapshot) { s.Weapon = "models/weapons/v_hyperb/tris.md2" }},
		{"only blaster and shotgun", "Blaster", 360, func(s *quake.Snapshot) { s.Inventory = s.Inventory[:3] }},
		{"unknown inventory", "", 360, func(s *quake.Snapshot) { s.InventoryKnown = false }},
		{"stale inventory", "", 360, func(s *quake.Snapshot) { s.InventoryAgeFrames = 21 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := economySnapshot()
			s.Weapon = "models/weapons/v_shotg/tris.md2"
			s.Enemies[0].Class = "monster_parasite"
			s.Enemies[0].Origin[0] = tc.distance
			tc.change(&s)
			got, _ := economyWeapon(s)
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
	s := economySnapshot()
	s.Weapon = "models/weapons/v_shotg/tris.md2"
	s.Enemies[0].Class = "monster_parasite"
	profile := combatSpacing(s)
	if !profile.NeedSpace || !profile.RangeConflict || profile.Minimum != 320 || profile.PreferredMax != 192 {
		t.Fatal("shotgun conflict concealed", profile)
	}
	w := weaponSwitch{}
	w.command(s)
	s.Frame += 2
	if got := w.command(s); got != "use Machinegun" || w.reason != "parasite_retreat_range" {
		t.Fatal("bounded switch not issued", got, w)
	}
}

func TestParasiteConstrainsWeaponInObservedMixedGroup(t *testing.T) {
	for _, tc := range []struct {
		name, class, want string
		parasiteDistance  float64
		visible           bool
	}{
		{"nearer infantry", "monster_infantry", "Machinegun", 168, true},
		{"higher ranked tank", "monster_tank", "Machinegun", 168, true},
		{"far parasite leaves close weapon choice", "monster_infantry", "Shotgun", 500, true},
		{"hidden parasite is not a weapon constraint", "monster_infantry", "Shotgun", 168, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := economySnapshot()
			s.Weapon = "Blaster"
			s.Enemies[0].Class = tc.class
			s.Enemies[0].Origin = quake.Vec3{64, 0, 24}
			s.Enemies = append(s.Enemies, quake.Object{ID: 2, Class: "monster_parasite", Origin: quake.Vec3{tc.parasiteDistance, 0, 24}, ClearShot: &tc.visible})
			got, _ := economyWeapon(s)
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestEconomySwitchStableBoundedAndEmptyPriority(t *testing.T) {
	s := economySnapshot()
	w := weaponSwitch{}
	for frame := 10; frame <= 135; frame++ {
		s.Frame = frame
		got := w.command(s)
		want := ""
		if frame == 12 || frame == 52 || frame == 92 {
			want = "use Shotgun"
		}
		if got != want {
			t.Fatalf("frame%d got %q want %q", frame, got, want)
		}
		if w.command(s) != "" {
			t.Fatal("duplicate frame switched")
		}
	}
	s.Frame = 136
	s.Ammo = 0
	w.command(s)
	s.Frame = 138
	if got := w.command(s); got == "" || w.reason != "empty_weapon" {
		t.Fatal("empty fallback blocked by economy cooldown")
	}
	s.Health = 0
	s.Frame = 139
	if w.command(s) != "" {
		t.Fatal("dead switch")
	}
	s = economySnapshot()
	s.Frame = 1
	w.command(s)
	s.Frame = 3
	if w.command(s) != "use Shotgun" {
		t.Fatal("respawn/frame reset retained cooldown")
	}
}

func TestParasiteDepletionPreservesRangeDespiteFreshStaleAmmoCount(t *testing.T) {
	for _, tc := range []struct {
		weapon, want   string
		bullets, cells int
	}{
		{"models/weapons/v_machn/tris.md2", "use HyperBlaster", 5, 8},
		{"models/weapons/v_hyperb/tris.md2", "use Blaster", 0, 8},
	} {
		s := economySnapshot()
		s.Enemies[0].Class = "monster_parasite"
		s.Enemies[0].Origin[0] = 168
		s.Weapon = tc.weapon
		s.Ammo = 0
		s.Inventory[6].Count = tc.bullets
		s.Inventory[8].Count = tc.cells
		w := weaponSwitch{economyAt: 9}
		for frame := 10; frame <= 12; frame++ {
			s.Frame = frame
			got := w.command(s)
			if frame == 12 && (got != tc.want || w.reason != "empty_weapon") {
				t.Fatalf("%s: %q (%s), want %q", tc.weapon, got, w.reason, tc.want)
			}
		}
		s.InventoryAgeFrames = 21
		s.Frame = 30
		if name, _ := economyWeapon(s); name != "" {
			t.Fatal("stale inventory used for range selection")
		}
	}
}

func TestParasiteEmptyFallbackDuringBriefOcclusion(t *testing.T) {
	s := economySnapshot()
	s.Enemies[0].Class = "monster_parasite"
	s.Enemies[0].Origin[0] = 168
	s.Weapon = "models/weapons/v_hyperb/tris.md2"
	s.Ammo = 1
	s.Inventory[6].Count = 0
	s.Inventory[8].Count = 8
	w := weaponSwitch{}
	s.Frame = 10
	w.command(s)
	s.Enemies = nil
	s.Ammo = 0
	for frame := 11; frame <= 13; frame++ {
		s.Frame = frame
		got := w.command(s)
		if frame == 13 && got != "use Blaster" {
			t.Fatal("brief loss of LOS restored shotgun", got)
		}
	}
	// No indefinite hidden-enemy constraint or state across maps/death.
	s.Inventory[3].Count = 0
	s.Frame = 40
	w = weaponSwitch{parasiteSeenAt: 10, mapName: s.Map, lastFrame: 39, emptyAt: 37}
	if got := w.command(s); got != "use Shotgun" {
		t.Fatal("expired constraint retained", got)
	}
	s.Map = "base2"
	s.Frame = 41
	w.command(s)
	if w.parasiteSeenAt != 0 {
		t.Fatal("constraint survived map change")
	}
	s.Health = 0
	s.Frame = 42
	w.parasiteSeenAt = 40
	w.command(s)
	if w.parasiteSeenAt != 0 {
		t.Fatal("constraint survived death")
	}
}
