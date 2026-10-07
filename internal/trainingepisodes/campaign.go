package trainingepisodes

import (
	"fmt"
	"math"

	"q2coopbot/internal/quake"
)

// CampaignSite pins a real map encounter to its original BSP entity and geometry.
// WallDistances are eight horizontal static rays at player eye height, not labels
// inferred from map names. Native movers are checked again by the live harness.
type CampaignSite struct {
	Map           string     `json:"map"`
	EntityIndex   int        `json:"entity_index"`
	SourceClass   string     `json:"source_class"`
	SourceOrigin  quake.Vec3 `json:"source_origin"`
	BSPSHA256     string     `json:"bsp_sha256"`
	WallDistances []float64  `json:"wall_distances"`
}

// CampaignRecipes selects spatially separated, grounded fights around original
// campaign encounters. It does not claim to reproduce an entire campaign state.
func CampaignRecipes(root string, perMap int) ([]Episode, error) {
	if perMap < 1 || perMap > 12 {
		return nil, fmt.Errorf("campaign sites per map must be 1..12")
	}
	var result []Episode
	for mapIndex, name := range []string{"base1", "base2"} {
		world, err := loadGenerationWorld(root, name)
		if err != nil {
			return nil, err
		}
		var selected []quake.Vec3
		for entityIndex, entity := range world.Entities {
			class := entity.Class
			switch class {
			case "monster_soldier_light", "monster_soldier_ss":
				class = "monster_soldier"
			case "monster_soldier", "monster_infantry", "monster_parasite":
			default:
				continue
			}
			far := true
			for _, p := range selected {
				if quake.Horizontal(p, entity.Origin) < 512 {
					far = false
				}
			}
			if !far {
				continue
			}
			ep, ok := campaignRecipe(world, entity, entityIndex, class, mapIndex, len(selected))
			if !ok {
				continue
			}
			selected = append(selected, entity.Origin)
			result = append(result, ep)
			if len(selected) == perMap {
				break
			}
		}
		if len(selected) != perMap {
			return nil, fmt.Errorf("%s: only %d supported distinct sites, requested %d", name, len(selected), perMap)
		}
	}
	return result, nil
}

func grounded(world *quake.MapInfo, p quake.Vec3) (quake.Vec3, bool) {
	p[2] += 48
	drop, ok := world.GroundDrop(p, 128)
	if !ok {
		return p, false
	}
	// GroundDrop's plane inset places the result .125 below the surface.
	// Add .25 to keep the actor feet .125 above it, matching native settling.
	p[2] = math.Round((p[2]-drop)*8)/8 + .25
	return p, true
}

func campaignRecipe(world quake.MapInfo, source quake.MapEntity, entityIndex int, class string, mapIndex, sceneIndex int) (Episode, bool) {
	primary, ok := grounded(&world, source.Origin)
	if !ok {
		return Episode{}, false
	}
	for _, distance := range []float64{192, 128, 256, 96} {
		for angle := 0; angle < 16; angle++ {
			a := float64(angle) * math.Pi / 8
			player := primary
			player[0] += distance * math.Cos(a)
			player[1] += distance * math.Sin(a)
			player, ok = grounded(&world, player)
			if !ok {
				continue
			}
			player[0] = math.Round(player[0]*8) / 8
			player[1] = math.Round(player[1]*8) / 8
			if checkStart(Instance{Map: world.Name, Player: player, Monsters: []GeneratedMonster{{Class: class, Position: primary}}}, &world) != "" {
				continue
			}
			id := fmt.Sprintf("campaign-%s-site-%02d-blaster", world.Name, sceneIndex+1)
			ep := Episode{Version: 1, ID: id, Revision: 1, Title: fmt.Sprintf("%s encounter %d: %s", world.Name, entityIndex, class), Status: "runnable", Map: world.Name, Scope: "first_life_combat", Goal: "alive_native_kills_then_observed_effect", Monsters: []string{class}, Geometry: "Original campaign BSP encounter; distinct location; static hull/floor/visibility verified; native doors and movers retained", InitialState: "Isolated fight at original encounter location; other campaign monsters removed; original soldier variants use shotgun Soldier; stock monster HP; generated player HP80/100", Skill: 1, Timescale: 2, GameFrames: 300, Modes: []string{"rules", "learned"}, PPOTrainable: true, Splits: map[string]Seeds{}, Recipe: Recipe{Runner: "combat-baseline", Loadout: "blaster", RewardConfig: "scripts/scenarios/combat-reward-recoil-v5.json"}, Missing: []string{}, Generator: &Generator{Version: 1, Kind: "campaign-ground-combat-v1", Distributions: map[string]StartDistribution{}}}
			site := &CampaignSite{Map: world.Name, EntityIndex: entityIndex, SourceClass: source.Class, SourceOrigin: source.Origin, BSPSHA256: world.BSPSHA256}
			eye := player
			eye[2] += 22
			for ray := 0; ray < 8; ray++ {
				end := eye
				end[0] += 512 * math.Cos(float64(ray)*math.Pi/4)
				end[1] += 512 * math.Sin(float64(ray)*math.Pi/4)
				tr := world.TraceProjectile(eye, end)
				site.WallDistances = append(site.WallDistances, math.Round(tr.Fraction*512))
			}
			ep.Generator.Site = site
			base := 1000000 + mapIndex*600000 + sceneIndex*40000
			for i, split := range splitNames {
				ep.Splits[split] = Seeds{Start: base + i*10000, Count: 4096}
				lo, hi := primary, primary
				lo[0] += float64(i*4 - 8)
				hi[0] = lo[0] + 2
				lo[1] -= 2
				hi[1] += 2
				pl, ph := player, player
				pl[0] -= 4
				pl[1] -= 4
				ph[0] += 4
				ph[1] += 4
				ep.Generator.Distributions[split] = StartDistribution{Player: PositionRange{Min: pl, Max: ph}, Primary: PositionRange{Min: lo, Max: hi}, Health: []int{80, 100}}
			}
			valid := ep.validate() == nil
			for _, split := range splitNames {
				for n := 0; n < 16 && valid; n++ {
					_, err := generate(ep, split, ep.Splits[split].Start+n, &world)
					valid = err == nil
				}
			}
			if valid {
				return ep, true
			}
		}
	}
	return Episode{}, false
}
