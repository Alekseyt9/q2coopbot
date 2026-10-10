package trainingepisodes

import (
	"fmt"
	"math"
	"strings"

	"q2coopbot/internal/quake"
)

// CampaignGroupRecipes prepares distinct-class groups around checked campaign sites.
// Native startup and infighting evidence must still be checked by the harness.
func CampaignGroupRecipes(root string, perMap int, maps, classes []string) ([]Episode, error) {
	return CampaignGroupRecipesForLoadout(root, perMap, maps, classes, "weapons-ssg")
}

// CampaignGroupRecipesForLoadout keeps legacy SSG seeds and reserves separate
// domains for other supported inventories. It never changes the policy contract.
func CampaignGroupRecipesForLoadout(root string, perMap int, maps, classes []string, loadout string) ([]Episode, error) {
	loadoutIndex := -1
	for i, name := range []string{"blaster", "machinegun", "weapons"} {
		if name == loadout {
			loadoutIndex = i
		}
	}
	if loadout != "weapons-ssg" && loadoutIndex < 0 {
		return nil, fmt.Errorf("unsupported group loadout")
	}
	order := []string{"monster_parasite", "monster_soldier", "monster_infantry", "monster_gunner"}
	selected := map[string]bool{}
	for _, class := range classes {
		if selected[class] {
			return nil, fmt.Errorf("duplicate group class")
		}
		selected[class] = true
	}
	var canonical []string
	mask := 0
	for i, class := range order {
		if selected[class] {
			canonical = append(canonical, class)
			mask |= 1 << i
			delete(selected, class)
		}
	}
	if len(selected) != 0 || len(canonical) < 2 || len(canonical) > 4 {
		return nil, fmt.Errorf("unsupported group composition")
	}
	parents, err := CampaignRecipesForMaps(root, perMap, maps)
	if err != nil {
		return nil, err
	}
	var result []Episode
	for index, parent := range parents {
		world, err := loadGenerationWorld(root, parent.Map)
		if err != nil {
			return nil, err
		}
		d := parent.Generator.Distributions["train"]
		center := func(box PositionRange) quake.Vec3 {
			return quake.Vec3{(box.Min[0] + box.Max[0]) / 2, (box.Min[1] + box.Max[1]) / 2, (box.Min[2] + box.Max[2]) / 2}
		}
		primary, player := center(d.Primary), center(d.Player)
		monsters := []GeneratedMonster{{Class: canonical[0], Position: primary}}
		var boxes []PositionRange
		for _, class := range canonical[1:] {
			found := false
			for _, radius := range []float64{64, 96, 128, 192, 256} {
				for angle := 0; angle < 16 && !found; angle++ {
					p := primary
					p[0] += radius * math.Cos(float64(angle)*math.Pi/8)
					p[1] += radius * math.Sin(float64(angle)*math.Pi/8)
					p, groundedOK := grounded(&world, p)
					if !groundedOK {
						continue
					}
					candidate := append(append([]GeneratedMonster(nil), monsters...), GeneratedMonster{Class: class, Position: p})
					if checkStart(Instance{Map: parent.Map, Player: player, Monsters: candidate}, &world) != "" {
						continue
					}
					monsters = candidate
					lo, hi := p, p
					lo[0]--
					lo[1]--
					hi[0]++
					hi[1]++
					boxes = append(boxes, PositionRange{Min: lo, Max: hi})
					found = true
				}
				if found {
					break
				}
			}
			if !found {
				return nil, fmt.Errorf("%s: no checked position for %s", parent.ID, class)
			}
		}
		labels := []string{}
		for _, class := range canonical {
			labels = append(labels, strings.TrimPrefix(class, "monster_"))
		}
		parent.ID = strings.TrimSuffix(parent.ID, "-blaster") + "-group-" + strings.Join(labels, "-") + "-" + loadout
		parent.Monsters = append([]string(nil), canonical...)
		parent.Title += " / group " + strings.Join(labels, ", ")
		parent.Recipe.Mixed, parent.Recipe.Loadout = true, loadout
		parent.Recipe.RewardConfig = "scripts/scenarios/combat-reward-selected-aim-v8.json"
		parent.Modes = []string{"learned"}
		parent.Generator.Kind = "campaign-ground-group-v1"
		parent.InitialState = "Isolated stock-health distinct-class ground group; Super Shotgun initial, MG/Shotgun/Blaster owned; native no-infighting fixture"
		if loadout != "weapons-ssg" {
			parent.InitialState = "Isolated stock-health distinct-class ground group; " + loadout + " inventory; native no-infighting fixture"
		}
		mapIndex := 0
		for i, name := range campaignMapOrder {
			if name == parent.Map {
				mapIndex = i
			}
		}
		for splitIndex, split := range splitNames {
			distribution := parent.Generator.Distributions[split]
			distribution.Additional = append([]PositionRange(nil), boxes...)
			parent.Generator.Distributions[split] = distribution
			parent.Splits[split] = Seeds{Start: 200000000 + mapIndex*16000000 + (index%perMap)*1000000 + mask*40000 + splitIndex*10000, Count: 4096}
			if loadoutIndex >= 0 {
				parent.Splits[split] = Seeds{Start: 1000000000 + loadoutIndex*80000000 + mapIndex*2000000 + (index%perMap)*100000 + mask*5000 + splitIndex*1000, Count: 512}
			}
		}
		if err := parent.validate(); err != nil {
			return nil, err
		}
		for _, split := range splitNames {
			for offset := 0; offset < 16; offset++ {
				if _, err := generate(parent, split, parent.Splits[split].Start+offset, &world); err != nil {
					return nil, err
				}
			}
		}
		result = append(result, parent)
	}
	return result, nil
}
