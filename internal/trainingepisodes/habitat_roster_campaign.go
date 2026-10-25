package trainingepisodes

import (
	"fmt"
	"math"
	"q2coopbot/internal/quake"
	"sort"
)

// NativeHabitatRosterRecipes adds class-specific placements to each admitted
// original native profile encounter. Count caps reflect geometry preflight,
// never combat outcomes; singleton recipes remain explicit when groups cannot fit.
func NativeHabitatRosterRecipes(root string, perType int) ([]Episode, []MapCoverage, error) {
	parents, coverage, err := NativeProfileRecipes(root, perType)
	if err != nil {
		return nil, nil, err
	}
	worlds := map[string]quake.MapInfo{}
	var result []Episode
	center := func(b PositionRange) quake.Vec3 {
		return quake.Vec3{(b.Min[0] + b.Max[0]) / 2, (b.Min[1] + b.Max[1]) / 2, (b.Min[2] + b.Max[2]) / 2}
	}
	for _, ep := range parents {
		world, ok := worlds[ep.Map]
		if !ok {
			world, err = loadGenerationWorld(root, ep.Map)
			if err != nil {
				return nil, nil, err
			}
			worlds[ep.Map] = world
		}
		primaryClass := ep.Monsters[0]
		base := ep.Generator.Distributions["train"]
		player, primary := center(base.Player), center(base.Primary)
		pool := []string{}
		for class := range nativeMonsterCounts(world) {
			if _, ok := monsterProfiles[class]; ok {
				pool = append(pool, class)
			}
		}
		sort.Strings(pool)
		placements := map[string][]PositionRange{}
		for _, class := range pool {
			profile := monsterProfiles[class]
			add := func(p quake.Vec3) {
				if len(placements[class]) >= 12 {
					return
				}
				if profile.Habitat == "ground" {
					var ok bool
					p, ok = groundForHull(&world, p, profile.Min[2], 512)
					if !ok {
						return
					}
				}
				for _, existing := range placements[class] {
					q := center(existing)
					if quake.Horizontal(p, q) < 40 && math.Abs(p[2]-q[2]) < 48 {
						return
					}
				}
				v := Instance{Map: ep.Map, MonsterGeometry: nativeProfileGeometry, Player: player, Monsters: []GeneratedMonster{{Class: primaryClass, Position: primary}, {Class: class, Position: p}}}
				eye, target := player, p
				eye[2] += 22
				target[2] += (profile.Min[2] + profile.Max[2]) / 2
				a, ak := world.StaticArea(eye)
				b, bk := world.StaticArea(target)
				if !ak || !bk || a != b {
					return
				}
				if checkStart(v, &world) != "" {
					return
				}
				lo, hi := p, p
				lo[0] -= .5
				lo[1] -= .5
				hi[0] += .5
				hi[1] += .5
				placements[class] = append(placements[class], PositionRange{Min: lo, Max: hi})
			}
			// Native habitat anchors are especially important for water and high air.
			for _, source := range world.Entities {
				if source.Class != class || quake.Distance(source.Origin, primary) > 768 {
					continue
				}
				add(source.Origin)
			}
			if profile.Habitat == "water" {
				for _, source := range world.Entities {
					if source.Class != class || quake.Distance(source.Origin, primary) > 768 {
						continue
					}
					for _, radius := range []float64{48, 80, 128} {
						for angle := 0; angle < 12; angle++ {
							p := source.Origin
							p[0] += radius * math.Cos(float64(angle)*math.Pi/6)
							p[1] += radius * math.Sin(float64(angle)*math.Pi/6)
							add(p)
						}
					}
				}
			} else {
				for _, radius := range []float64{80, 128, 192, 256, 384, 512} {
					for angle := 0; angle < 16; angle++ {
						if len(placements[class]) >= 12 {
							break
						}
						p := primary
						p[0] += radius * math.Cos(float64(angle)*math.Pi/8)
						p[1] += radius * math.Sin(float64(angle)*math.Pi/8)
						if profile.Habitat == "ground" {
							p[2] = math.Max(primary[2], player[2]) + 96
							add(p)
						} else {
							for _, z := range []float64{player[2] + 64, player[2] + 128, primary[2] + 32} {
								p[2] = z
								add(p)
							}
						}
					}
				}
			}
		}
		pool = nil
		for class, boxes := range placements {
			if len(boxes) > 0 {
				pool = append(pool, class)
			}
		}
		sort.Strings(pool)
		ep.ID += "-random-roster"
		ep.Title += " / randomized native habitat group"
		ep.Generator.Kind = nativeHabitatRosterKind
		ep.Generator.Roster = &MonsterDistribution{Version: 2, PrimaryClass: primaryClass, MinCount: 2, MaxCount: 4, ClassPool: pool}
		if len(pool) == 0 {
			ep.Generator.Roster.MinCount = 1
			ep.Generator.Roster.MaxCount = 1
		}
		ep.Monsters = habitatRosterClasses(*ep.Generator.Roster)
		ep.InitialState = "Fixed original native primary profile; sampled native additional classes with replacement and per-class habitat placements; partial stock inventory"
		for _, split := range splitNames {
			d := ep.Generator.Distributions[split]
			d.AdditionalByClass = placements
			ep.Generator.Distributions[split] = d
			s := ep.Splits[split]
			s.Start -= 180000000
			ep.Splits[split] = s
		}
		// Reduce only the spatial count cap when maximum-size sampled groups do
		// not fit. Class sampling is not retried based on model/game outcomes.
		for {
			ep.Recipe.Mixed = ep.Generator.Roster.MaxCount > 1
			if err := ep.validate(); err != nil {
				return nil, nil, err
			}
			valid := true
			for _, split := range splitNames {
				for offset := 0; offset < 16; offset++ {
					if _, err := generate(ep, split, ep.Splits[split].Start+offset, &world); err != nil {
						valid = false
						break
					}
				}
				if !valid {
					break
				}
			}
			if valid {
				result = append(result, ep)
				break
			}
			if ep.Generator.Roster.MaxCount == 1 {
				return nil, nil, fmt.Errorf("%s: singleton habitat preflight exhausted", ep.ID)
			}
			if ep.Generator.Roster.MaxCount <= 2 {
				ep.Generator.Roster.MaxCount = 1
				ep.Generator.Roster.MinCount = 1
				ep.Generator.Roster.ClassPool = nil
				ep.Monsters = []string{primaryClass}
				for _, split := range splitNames {
					d := ep.Generator.Distributions[split]
					d.AdditionalByClass = nil
					ep.Generator.Distributions[split] = d
				}
			} else {
				ep.Generator.Roster.MaxCount--
			}
		}
	}
	return result, coverage, nil
}
