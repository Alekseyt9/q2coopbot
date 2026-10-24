package trainingepisodes

import (
	"math/bits"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
)

func TestProjectileGroupSeedDomains(t *testing.T) {
	// Pinned historical base3/site01 parasite-infantry-gunner weapons recipe.
	if got := campaignGroupSeeds(2, 0, 13, 2, 0); got != (Seeds{Start: 1164065000, Count: 512}) {
		t.Fatalf("historical registered seed domain changed: %+v", got)
	}
	type domain struct {
		Seeds
		projectile bool
	}
	var domains []domain
	for mapIndex := range campaignMapOrder {
		for site := 0; site < 12; site++ {
			for loadout := -1; loadout < 4; loadout++ {
				for mask := 1; mask < 32; mask++ {
					if n := bits.OnesCount(uint(mask)); n < 2 || n > 4 {
						continue
					}
					for split := range splitNames {
						d := campaignGroupSeeds(mapIndex, site, mask, loadout, split)
						if d.Start < 0 || int64(d.Start)+int64(d.Count) > 2147483648 {
							t.Fatal("engine seed overflow", d)
						}
						domains = append(domains, domain{d, mask&16 != 0})
					}
				}
			}
		}
	}
	sort.Slice(domains, func(i, j int) bool { return domains[i].Start < domains[j].Start })
	for i := 1; i < len(domains); i++ {
		previous, current := domains[i-1], domains[i]
		if previous.Start+previous.Count > current.Start && (previous.projectile || current.projectile) {
			t.Fatalf("new seed domain overlaps another registered family: %+v %+v", previous, current)
		}
	}
}

func TestProjectileGroupPreservesExactLightSoldier(t *testing.T) {
	root, _ := filepath.Abs("../..")
	world, err := loadGenerationWorld(root, "base3")
	if err != nil {
		t.Fatal(err)
	}
	for _, classes := range [][]string{
		{"monster_gunner", "monster_soldier_light"},
		{"monster_gunner", "monster_soldier_light", "monster_parasite"},
	} {
		episodes, err := CampaignGroupRecipesForLoadout(root, 1, []string{"base3"}, classes, "weapons")
		if err != nil {
			t.Fatal(err)
		}
		ep := episodes[0]
		if ep.Generator.Kind != projectileGroupKind || ep.Monsters[0] == "monster_gunner" {
			t.Fatal("exact group lost kind or supported primary", ep.Monsters)
		}
		legacy := ep
		copyGenerator := *ep.Generator
		legacy.Generator = &copyGenerator
		legacy.Generator.Kind = "campaign-ground-group-v1"
		if legacy.validate() == nil {
			t.Fatal("historical generator silently accepted a new class")
		}
		for _, split := range splitNames {
			instance, err := generate(ep, split, ep.Splits[split].Start, &world)
			if err != nil {
				t.Fatal(err)
			}
			var generated []string
			for _, monster := range instance.Monsters {
				generated = append(generated, monster.Class)
			}
			if !reflect.DeepEqual(generated, ep.Monsters) || checkStart(instance, &world) != "" {
				t.Fatal("wrong native composition or unsafe static placement", generated)
			}
		}
	}
}
