package trainingepisodes

import (
	"fmt"
	"sort"
)

// NativeProfileRecipes scans original anchors separately for every combat type.
// It never replaces an unsupported/passive type with a different enemy.
func NativeProfileRecipes(root string, perType int) ([]Episode, []MapCoverage, error) {
	if perType < 1 || perType > 12 {
		return nil, nil, fmt.Errorf("profile sites per type must be1..12")
	}
	keys := []string{}
	for class := range monsterProfiles {
		keys = append(keys, class)
	}
	sort.Strings(keys)
	var episodes []Episode
	var coverage []MapCoverage
	for mapIndex, name := range AllCombatMapNames() {
		entry := MapCoverage{Map: name, Kind: "campaign", State: "blocked", ProfileSites: map[string]int{}, ProfileMissing: map[string]string{}}
		world, err := loadGenerationWorld(root, name)
		if err != nil {
			entry.Reason = err.Error()
			coverage = append(coverage, entry)
			continue
		}
		entry.GeometrySHA256 = world.BSPSHA256
		entry.NativeMonsters = nativeMonsterCounts(world)
		for class := range entry.NativeMonsters {
			if _, ok := monsterProfiles[class]; !ok {
				entry.ProfileMissing[class] = "Not an admitted stock combat profile; no replacement"
			}
		}
		for classIndex, class := range keys {
			if entry.NativeMonsters[class] == 0 {
				continue
			}
			for entityIndex, source := range world.Entities {
				if source.Class != class {
					continue
				}
				sceneIndex := classIndex*perType + entry.ProfileSites[class]
				ep, ok := campaignRecipe(world, source, entityIndex, class, mapIndex, sceneIndex, nativeProfileKind)
				if !ok {
					continue
				}
				episodes = append(episodes, ep)
				entry.Sites++
				entry.ProfileSites[class]++
				if entry.ProfileSites[class] == perType {
					break
				}
			}
			if entry.ProfileSites[class] == 0 {
				entry.ProfileMissing[class] = "Original anchors failed profile hull/habitat/player-floor/visibility preflight"
			}
		}
		if entry.Sites > 0 {
			entry.State = "geometry_preflight_passed"
		} else {
			entry.Reason = "No original native combat anchor passed per-type profile preflight"
		}
		coverage = append(coverage, entry)
	}
	return episodes, coverage, nil
}
