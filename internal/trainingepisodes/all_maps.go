package trainingepisodes

import (
	"fmt"
	"strings"

	"q2coopbot/internal/quake"
)

type MapCoverage struct {
	ProfileSites   map[string]int    `json:"profile_sites,omitempty"`
	ProfileMissing map[string]string `json:"profile_missing,omitempty"`
	Map            string            `json:"map"`
	Kind           string            `json:"kind"`
	State          string            `json:"state"`
	Sites          int               `json:"sites"`
	Reason         string            `json:"reason,omitempty"`
	GeometrySHA256 string            `json:"geometry_sha256,omitempty"`
	NativeVerified bool              `json:"native_verified"`
	NativeMonsters map[string]int    `json:"native_monsters"`
}

// AllMapRecipes includes every stock campaign map, using real grounded BSP anchors when
// a map has no supported authored monster. It is geometric preflight, not a
// declaration that native spawning or campaign completion has been tested.
func AllMapRecipes(root string, perMap int) ([]Episode, []MapCoverage, error) {
	if perMap < 1 || perMap > 12 {
		return nil, nil, fmt.Errorf("map sites must be 1..12")
	}
	var episodes []Episode
	var coverage []MapCoverage
	for index, name := range AllCombatMapNames() {
		entry := MapCoverage{Map: name, Kind: "campaign", State: "blocked"}
		world, err := loadGenerationWorld(root, name)
		if err != nil {
			entry.Reason = err.Error()
			coverage = append(coverage, entry)
			continue
		}
		entry.GeometrySHA256 = world.BSPSHA256
		entry.NativeMonsters = nativeMonsterCounts(world)
		fallbackClass := nativeGroundClass(entry.NativeMonsters)
		if fallbackClass == "" {
			entry.Reason = "Native monster pool is empty or requires unsupported hull/habitat; no foreign replacement is permitted"
			coverage = append(coverage, entry)
			continue
		}
		selected := []quake.Vec3{}
		// Supported monster anchors first, then player starts and ordinary
		// item locations. No random points in walls, air or moving brushes.
		for pass := 0; pass < 3 && entry.Sites < perMap; pass++ {
			for entityIndex, entity := range world.Entities {
				class, eligible := siteAnchorClass(entity.Class, pass)
				if !eligible {
					continue
				}
				if pass > 0 {
					class = fallbackClass
				}
				if entry.NativeMonsters[class] == 0 {
					continue
				}
				far := true
				for _, previous := range selected {
					if quake.Horizontal(previous, entity.Origin) < 512 {
						far = false
						break
					}
				}
				if !far {
					continue
				}
				ep, ok := campaignRecipe(world, entity, entityIndex, class, index, entry.Sites, nativeGroundKind)
				if !ok {
					continue
				}
				if pass > 0 {
					ep.Geometry = "Original BSP grounded player/item anchor; static hull, floor, visibility and seed distributions verified; native spawning still requires a live check"
					ep.InitialState = "Isolated generated ground encounter at a real map anchor; exact monster class selected only from this map's original BSP; player HP80/100"
				}
				episodes = append(episodes, ep)
				selected = append(selected, entity.Origin)
				entry.Sites++
				if entry.Sites == perMap {
					break
				}
			}
		}
		if entry.Sites > 0 {
			entry.State = "geometry_preflight_passed"
		} else {
			entry.Reason = "No grounded supported encounter or player/item anchor passed hull/floor/visibility validation"
		}
		coverage = append(coverage, entry)
	}
	return episodes, coverage, nil
}

func siteAnchorClass(class string, pass int) (string, bool) {
	if pass == 0 {
		switch class {
		case "monster_soldier_light", "monster_soldier", "monster_soldier_ss", "monster_gunner", "monster_infantry", "monster_parasite", "monster_berserk", "monster_brain":
			return class, true
		}
	}
	if pass == 1 && (class == "info_player_start" || class == "info_player_coop") {
		return "monster_soldier", true
	}
	if pass == 2 && (strings.HasPrefix(class, "ammo_") || strings.HasPrefix(class, "weapon_") || class == "item_health" || class == "item_health_large") {
		return "monster_soldier", true
	}
	return "", false
}

const nativeGroundKind = "campaign-native-ground-combat-v2"

func nativeMonsterCounts(world quake.MapInfo) map[string]int {
	result := map[string]int{}
	for _, entity := range world.Entities {
		if strings.HasPrefix(entity.Class, "monster_") {
			result[entity.Class]++
		}
	}
	return result
}

func nativeGroundClass(pool map[string]int) string {
	for _, class := range []string{"monster_infantry", "monster_parasite", "monster_soldier", "monster_soldier_light", "monster_gunner", "monster_soldier_ss", "monster_berserk", "monster_brain"} {
		if pool[class] > 0 {
			return class
		}
	}
	return ""
}
