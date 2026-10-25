package trainingepisodes

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

func DefaultPartialInventoryDistribution() InventoryDistribution {
	return InventoryDistribution{Version: 1, WeaponPool: policy.WeaponChoices()[2:], MinWeapons: 0, MaxWeapons: 5, AmmoCounts: map[string][]int{
		"Bullets": {0, 1, 5, 20, 100}, "Shells": {0, 1, 2, 5, 20}, "Cells": {0, 1, 49, 50, 100},
		"Grenades": {0, 1, 3, 5}, "Rockets": {0, 1, 3, 5}, "Slugs": {0, 1, 3, 5},
	}}
}

// NativeRosterRecipes reserves checked positions for the maximum group size.
// Classes/counts are then sampled independently for each seed, with replacement.
// All currently admitted types fit [-16,-16,-24]..[16,16,32] in stock native
// source (Parasite is shorter). Larger/flying/swimming types need other geometry.
func NativeRosterRecipes(root string, perMap, minCount, maxCount int) ([]Episode, []MapCoverage, error) {
	d := MonsterDistribution{Version: 1, MinCount: minCount, MaxCount: maxCount, ClassPool: []string{"monster_infantry"}}
	if err := d.validate(); err != nil {
		return nil, nil, err
	}
	parents, coverage, err := AllMapRecipes(root, perMap)
	if err != nil {
		return nil, nil, err
	}
	var result []Episode
	counts := map[string]int{}
	for _, parent := range parents {
		world, err := loadGenerationWorld(root, parent.Map)
		if err != nil {
			return nil, nil, err
		}
		pool := []string{}
		for class := range nativeMonsterCounts(world) {
			if nativeRosterClass(class) {
				pool = append(pool, class)
			}
		}
		sort.Strings(pool)
		center := func(b PositionRange) quake.Vec3 {
			return quake.Vec3{(b.Min[0] + b.Max[0]) / 2, (b.Min[1] + b.Max[1]) / 2, (b.Min[2] + b.Max[2]) / 2}
		}
		primary := center(parent.Generator.Distributions["train"].Primary)
		player := center(parent.Generator.Distributions["train"].Player)
		monsters := []GeneratedMonster{{Class: pool[0], Position: primary}}
		var boxes []PositionRange
		for slot := 1; slot < maxCount; slot++ {
			found := false
			for _, radius := range []float64{64, 96, 128, 192, 256, 320} {
				for angle := 0; angle < 16 && !found; angle++ {
					p := primary
					p[0] += radius * math.Cos(float64(angle)*math.Pi/8)
					p[1] += radius * math.Sin(float64(angle)*math.Pi/8)
					p, ok := grounded(&world, p)
					if !ok {
						continue
					}
					candidate := append(append([]GeneratedMonster(nil), monsters...), GeneratedMonster{Class: pool[0], Position: p})
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
				break
			}
		}
		if len(boxes) != maxCount-1 {
			continue
		}
		parent.ID = strings.TrimSuffix(parent.ID, "-blaster") + "-random-roster-inventory"
		parent.Title = fmt.Sprintf("%s native group of %d..%d with partial inventory", parent.Map, minCount, maxCount)
		parent.Monsters = pool
		parent.Recipe.Mixed, parent.Recipe.Loadout = true, "weapons"
		parent.InitialState = "Exact original BSP class pool; sampled with replacement, stock health, partial stock weapon/ammo inventory; no-infighting fixture"
		parent.Generator.Kind = nativeRosterKind
		parent.Generator.Roster = &MonsterDistribution{Version: 1, MinCount: minCount, MaxCount: maxCount, ClassPool: pool}
		inventory := DefaultPartialInventoryDistribution()
		parent.Generator.Inventory = &inventory
		for _, split := range splitNames {
			v := parent.Generator.Distributions[split]
			v.Additional = append([]PositionRange(nil), boxes...)
			parent.Generator.Distributions[split] = v
			// Disjoint from the80M fixed native family and200M legacy groups.
			parent.Splits[split] = Seeds{Start: parent.Splits[split].Start - 80000000 + 170000000, Count: 4096}
		}
		if err := parent.validate(); err != nil {
			return nil, nil, err
		}
		valid := true
		for _, split := range splitNames {
			for offset := 0; offset < 16; offset++ {
				if _, err := generate(parent, split, parent.Splits[split].Start+offset, &world); err != nil {
					valid = false
				}
			}
		}
		if !valid {
			continue
		}
		counts[parent.Map]++
		result = append(result, parent)
	}
	for i := range coverage {
		coverage[i].Sites = counts[coverage[i].Map]
		if coverage[i].Sites > 0 {
			coverage[i].State = "geometry_preflight_passed"
			coverage[i].Reason = ""
		} else if coverage[i].Reason == "" {
			coverage[i].State = "blocked"
			coverage[i].Reason = "No group positions passed maximum-count geometry preflight"
		}
	}
	return result, coverage, nil
}
