package trainingepisodes

import (
	"q2coopbot/internal/quake"
	"strings"
	"testing"
)

func TestNativeMonsterPoolNeverSubstitutesSoldierVariants(t *testing.T) {
	world := quake.MapInfo{Entities: []quake.MapEntity{{Class: "monster_soldier_light"}, {Class: "monster_soldier_ss"}, {Class: "monster_tank"}, {Class: "info_player_start"}}}
	pool := nativeMonsterCounts(world)
	if len(pool) != 3 || pool["monster_soldier"] != 0 || pool["monster_soldier_light"] != 1 || pool["monster_soldier_ss"] != 1 {
		t.Fatal("native variants changed", pool)
	}
	if class, ok := siteAnchorClass("monster_soldier_light", 0); !ok || class != "monster_soldier_light" {
		t.Fatal("light soldier substituted")
	}
	if nativeGroundClass(map[string]int{"monster_tank": 1}) != "" || nativeGroundClass(map[string]int{}) != "" {
		t.Fatal("foreign fallback for unsupported/empty pool")
	}
	ep := Episode{Map: "base1", Monsters: []string{"monster_soldier"}, Generator: &Generator{Kind: nativeGroundKind}}
	if _, err := generate(ep, "train", 1, &world); err == nil {
		t.Fatal("foreign monster accepted")
	}
}

func TestAllMapsPreserveLegacySeedsAndSeparateAnchors(t *testing.T) {
	all := AllCombatMapNames()
	if len(all) != 39 {
		t.Fatal("missing stock maps", len(all))
	}
	seen := map[string]bool{}
	for i, name := range all {
		if strings.HasPrefix(name, "q2dm") {
			t.Fatal("deathmatch map included", name)
		}
		if seen[name] {
			t.Fatal("duplicate map", name)
		}
		seen[name] = true
		if i < len(campaignMapOrder) && name != campaignMapOrder[i] {
			t.Fatal("legacy seed order changed")
		}
	}
	all[0] = "changed"
	if AllCombatMapNames()[0] != "base1" {
		t.Fatal("map list aliases internal state")
	}
	if _, ok := siteAnchorClass("trigger_hurt", 2); ok {
		t.Fatal("hazard accepted as item anchor")
	}
	if _, ok := siteAnchorClass("info_player_deathmatch", 0); ok {
		t.Fatal("player anchor changed legacy monster pass")
	}
	if _, ok := siteAnchorClass("info_player_deathmatch", 1); ok {
		t.Fatal("deathmatch anchor included")
	}
	if class, ok := siteAnchorClass("info_player_coop", 1); !ok || class != "monster_soldier" {
		t.Fatal("co-op anchor missing")
	}
}
