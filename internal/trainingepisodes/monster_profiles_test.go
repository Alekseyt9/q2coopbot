package trainingepisodes

import (
	"path/filepath"
	"q2coopbot/internal/quake"
	"testing"
)

func TestProfilePreflightRejectsObservedBadNativeStarts(t *testing.T) {
	root, _ := filepath.Abs("../..")
	for _, tc := range []struct {
		mapName, class  string
		player, monster quake.Vec3
	}{
		{"base1", "monster_soldier_light", quake.Vec3{-540, 685.5, -23.875}, quake.Vec3{-621, 865.25, -23.875}},
		{"base2", "monster_flyer", quake.Vec3{966.375, 511, 216.125}, quake.Vec3{769.75, 512.625, 184}},
		{"biggun", "monster_flyer", quake.Vec3{1442.125, 434.75, 32.125}, quake.Vec3{1433, 528, 48}},
		{"city3", "monster_chick", quake.Vec3{-11.25, -130.875, -631.875}, quake.Vec3{-206.125, -129.75, -655.875}},
	} {
		world, err := loadGenerationWorld(root, tc.mapName)
		if err != nil {
			t.Skip(err)
		}
		v := Instance{Map: tc.mapName, MonsterGeometry: nativeProfileGeometry, Player: tc.player, Monsters: []GeneratedMonster{{Class: tc.class, Position: tc.monster}}}
		if reason := checkStart(v, &world); reason == "" {
			t.Fatal("known invalid native start accepted", tc.mapName)
		}
	}
}

func TestStockMonsterProfilesUseSpawnHullsAndHabitats(t *testing.T) {
	for _, tc := range []struct {
		class    string
		min, max quake.Vec3
		hp       int
		habitat  string
	}{
		{"monster_tank", quake.Vec3{-32, -32, -16}, quake.Vec3{32, 32, 72}, 750, "ground"},
		{"monster_chick", quake.Vec3{-16, -16, 0}, quake.Vec3{16, 16, 56}, 175, "ground"},
		{"monster_jorg", quake.Vec3{-80, -80, 0}, quake.Vec3{80, 80, 140}, 3000, "ground"},
		{"monster_boss2", quake.Vec3{-56, -56, 0}, quake.Vec3{56, 56, 80}, 2000, "air"},
		{"monster_flipper", quake.Vec3{-16, -16, 0}, quake.Vec3{16, 16, 32}, 50, "water"},
	} {
		p := monsterProfiles[tc.class]
		if p.Min != tc.min || p.Max != tc.max || p.Health != tc.hp || p.Habitat != tc.habitat {
			t.Fatal(tc.class, p)
		}
	}
	for _, class := range []string{"monster_boss3_stand", "monster_insane", "monster_commander_body"} {
		if _, ok := monsterProfiles[class]; ok {
			t.Fatal("passive scenery admitted as combat", class)
		}
	}
	lo, hi := profileBounds(quake.Vec3{0, 0, 16.125}, monsterProfiles["monster_tank"])
	if lo[2] != .125 || hi[2] != 88.125 {
		t.Fatal("tank feet incorrectly treated as player feet", lo, hi)
	}
	if !boundsOverlap(lo, hi, quake.Vec3{24, -8, 0}, quake.Vec3{40, 8, 32}, 0) {
		t.Fatal("large hull overlap missed")
	}
	if boundsOverlap(lo, hi, quake.Vec3{40, -8, 0}, quake.Vec3{56, 8, 32}, 0) {
		t.Fatal("separate hulls collide")
	}
}

func TestProfilePreflightReservesRaisedPlayerAgainstFlyingActor(t *testing.T) {
	root, _ := filepath.Abs("../..")
	world, err := loadGenerationWorld(root, "jail3")
	if err != nil {
		t.Skip(err)
	}
	v := Instance{Map: "jail3", MonsterGeometry: nativeProfileGeometry, Player: quake.Vec3{-473.125, 334.5, 600.125},
		Monsters: []GeneratedMonster{
			{Class: "monster_berserk", Position: quake.Vec3{-287.625, 336.375, 600.125}},
			{Class: "monster_hover", Position: quake.Vec3{-478.625, 336.375, 664.125}},
		}}
	lo, hi := profileBounds(v.Player, MonsterProfile{Min: quake.Vec3{-16, -16, -24}, Max: quake.Vec3{16, 16, 32}})
	a, b := profileBounds(v.Monsters[1].Position, monsterProfiles["monster_hover"])
	if boundsOverlap(lo, hi, a, b, 8) {
		t.Fatal("Regression fixture must clear the settled player")
	}
	hi[2] += 10
	if !boundsOverlap(lo, hi, a, b, 0) {
		t.Fatal("Regression fixture must intersect native teleport KillBox")
	}
	if reason := checkStart(v, &world); reason != "profile_player_teleport_overlap" {
		t.Fatal("Known pre-release telefrag accepted", reason)
	}
}
