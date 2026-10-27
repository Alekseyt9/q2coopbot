package trainingepisodes

import (
	"fmt"
	"math"
	"sort"
)

// This is a pre-outcome curriculum heuristic, not a solvability certificate.
const DifficultyVersion = "combat_resource_difficulty_v1"
const BeginnerDifficultyPolicy = "beginner_resources_v1"

type EncounterDifficulty struct {
	Version                     string         `json:"version"`
	EpisodeID                   string         `json:"episode_id"`
	Seed                        int            `json:"seed"`
	Map                         string         `json:"map"`
	Tier                        string         `json:"tier"`
	EligibleInitialTraining     bool           `json:"eligible_initial_training"`
	EstimateOnly                bool           `json:"estimate_only"`
	Reasons                     []string       `json:"reasons"`
	MonsterCount                int            `json:"monster_count"`
	StockMonsterHP              int            `json:"stock_monster_hp"`
	ThreatWeightedHP            float64        `json:"threat_weighted_hp"`
	PlayerHealth                int            `json:"player_health"`
	Skill                       int            `json:"skill"`
	UsableWeapons               []string       `json:"usable_weapons"`
	Ammo                        map[string]int `json:"ammo"`
	ExpectedFiniteDamage        float64        `json:"expected_finite_damage"`
	ExpectedNonExplosiveDamage  float64        `json:"expected_nonexplosive_damage"`
	NonExplosiveCoverage        float64        `json:"nonexplosive_coverage"`
	ExpectedClearSeconds        float64        `json:"expected_clear_seconds"`
	ExpectedNonExplosiveSeconds float64        `json:"expected_nonexplosive_seconds"`
	WindowSeconds               float64        `json:"window_seconds"`
	NearestEnemyDistance        float64        `json:"nearest_enemy_distance"`
	FarthestEnemyDistance       float64        `json:"farthest_enemy_distance"`
	CrossfireDegrees            float64        `json:"crossfire_degrees"`
	GeometryChecked             bool           `json:"geometry_checked"`
	StartVisibleMonsters        int            `json:"start_visible_monsters"`
	BlockedMonsterIndices       []int          `json:"blocked_monster_indices,omitempty"`
}

type DifficultyReport struct {
	Version            string                `json:"version"`
	OutcomeIndependent bool                  `json:"outcome_independent"`
	Counts             map[string]int        `json:"counts"`
	Cases              []EncounterDifficulty `json:"cases"`
}

type budgetWeapon struct {
	name, ammo   string
	cost         int
	damage, rate float64
	explosive    bool
}

// Expected effective damage per shot and approximate shots/second. Accuracy
// discounts are deliberately explicit estimates. No splash/piercing multi-kill
// credit, quad damage, pickups or enemy infighting is assumed.
var budgetWeapons = []budgetWeapon{
	{"Shotgun", "Shells", 1, 24, 1, false},
	{"Super Shotgun", "Shells", 2, 54, .9, false},
	{"Machinegun", "Bullets", 1, 4.4, 10, false},
	{"Chaingun", "Bullets", 1, 4, 20, false},
	{"Grenade Launcher", "Grenades", 1, 48, 1, true},
	{"Rocket Launcher", "Rockets", 1, 45, 1, true},
	{"HyperBlaster", "Cells", 1, 12, 10, false},
	{"Railgun", "Slugs", 1, 75, .65, false},
	{"BFG10K", "Cells", 50, 80, .5, true},
	{"Grenades", "Grenades", 1, 37.5, .5, true},
}

func copyAmmo(items map[string]int) map[string]int {
	result := map[string]int{}
	for k, v := range items {
		result[k] = v
	}
	return result
}

func finiteBudget(weapons []budgetWeapon, ammo map[string]int) float64 {
	order := append([]budgetWeapon(nil), weapons...)
	sort.SliceStable(order, func(i, j int) bool {
		return order[i].damage/float64(order[i].cost) > order[j].damage/float64(order[j].cost)
	})
	left := copyAmmo(ammo)
	damage := 0.
	for _, w := range order {
		shots := left[w.ammo] / w.cost
		damage += float64(shots) * w.damage
		left[w.ammo] -= shots * w.cost
	}
	return damage
}

func budgetTime(hp float64, weapons []budgetWeapon, ammo map[string]int) float64 {
	order := append([]budgetWeapon(nil), weapons...)
	sort.SliceStable(order, func(i, j int) bool { return order[i].damage*order[i].rate > order[j].damage*order[j].rate })
	left := copyAmmo(ammo)
	seconds := 0.
	for _, w := range order {
		if hp <= 0 {
			break
		}
		shots := math.Min(float64(left[w.ammo]/w.cost), math.Ceil(hp/w.damage))
		if shots <= 0 {
			continue
		}
		seconds += .6 + shots/w.rate // allowance for selecting another weapon
		hp -= shots * w.damage
		left[w.ammo] -= int(shots) * w.cost
	}
	// Blaster is unlimited ammunition, not unlimited damage within a deadline.
	// Effective 8 HP/s includes missed/blocked shots; this is a planning estimate.
	if hp > 0 {
		seconds += .6 + hp/8
	}
	return seconds
}

func AssessEncounter(v Instance, gameFrames int) EncounterDifficulty {
	a := EncounterDifficulty{Version: DifficultyVersion, EpisodeID: v.EpisodeID, Seed: v.EngineSeed, Map: v.Map,
		Tier: "unassessed", EstimateOnly: true, Reasons: []string{}, MonsterCount: len(v.Monsters), PlayerHealth: v.Health, Skill: v.Skill,
		Ammo: map[string]int{}, UsableWeapons: []string{}, WindowSeconds: float64(gameFrames) / 10}
	if v.Inventory == nil || len(v.Monsters) == 0 || gameFrames <= 0 || v.Skill < 0 || v.Skill > 3 {
		a.Reasons = append(a.Reasons, "missing_inventory_monsters_or_time_window")
		return a
	}
	counts := map[string]int{}
	for _, item := range v.Inventory.Items {
		if item.Count < 0 {
			a.Reasons = append(a.Reasons, "invalid_inventory")
			return a
		}
		if _, exists := counts[item.Name]; exists {
			a.Reasons = append(a.Reasons, "duplicate_inventory_item")
			return a
		}
		counts[item.Name] = item.Count
	}
	if counts["Blaster"] < 1 {
		a.Reasons = append(a.Reasons, "missing_blaster_fallback")
		return a
	}
	for _, name := range inventoryAmmoOrder {
		a.Ammo[name] = counts[name]
	}
	a.UsableWeapons = append(a.UsableWeapons, "Blaster")
	var weapons, safe []budgetWeapon
	for _, w := range budgetWeapons {
		if counts[w.name] > 0 && counts[w.ammo] >= w.cost {
			weapons = append(weapons, w)
			a.UsableWeapons = append(a.UsableWeapons, w.name)
			if !w.explosive {
				safe = append(safe, w)
			}
		}
	}
	weights := map[string]float64{"monster_gunner": 1.35, "monster_gladiator": 1.3, "monster_tank": 1.3,
		"monster_tank_commander": 1.3, "monster_medic": 1.5, "monster_parasite": 1.2, "monster_mutant": 1.2,
		"monster_berserk": 1.1, "monster_brain": 1.4, "monster_flyer": 1.2, "monster_floater": 1.3}
	a.NearestEnemyDistance = math.MaxFloat64
	special := false
	for _, m := range v.Monsters {
		profile, ok := monsterProfiles[m.Class]
		if !ok {
			a.NearestEnemyDistance = 0
			a.Reasons = append(a.Reasons, "unknown_monster_profile:"+m.Class)
			return a
		}
		a.StockMonsterHP += profile.Health
		weight := weights[m.Class]
		if weight == 0 {
			weight = 1
		}
		a.ThreatWeightedHP += float64(profile.Health) * weight
		if profile.Health >= 750 || profile.Habitat == "water" || m.Class == "monster_medic" || m.Class == "monster_brain" {
			special = true
		}
		d := 0.
		for axis := range m.Position {
			delta := m.Position[axis] - v.Player[axis]
			d += delta * delta
		}
		d = math.Sqrt(d)
		a.NearestEnemyDistance = math.Min(a.NearestEnemyDistance, d)
		a.FarthestEnemyDistance = math.Max(a.FarthestEnemyDistance, d)
	}
	a.ThreatWeightedHP *= 1 + .15*float64(v.Skill)
	for i, left := range v.Monsters {
		for _, right := range v.Monsters[i+1:] {
			lx, ly := left.Position[0]-v.Player[0], left.Position[1]-v.Player[1]
			rx, ry := right.Position[0]-v.Player[0], right.Position[1]-v.Player[1]
			denom := math.Hypot(lx, ly) * math.Hypot(rx, ry)
			if denom > 0 {
				angle := math.Acos(math.Max(-1, math.Min(1, (lx*rx+ly*ry)/denom))) * 180 / math.Pi
				a.CrossfireDegrees = math.Max(a.CrossfireDegrees, angle)
			}
		}
	}
	a.ExpectedFiniteDamage = finiteBudget(weapons, a.Ammo)
	a.ExpectedNonExplosiveDamage = finiteBudget(safe, a.Ammo)
	a.NonExplosiveCoverage = a.ExpectedNonExplosiveDamage / a.ThreatWeightedHP
	a.ExpectedClearSeconds = budgetTime(a.ThreatWeightedHP, weapons, a.Ammo)
	a.ExpectedNonExplosiveSeconds = budgetTime(a.ThreatWeightedHP, safe, a.Ammo)
	if len(v.Monsters) > 2 {
		a.Reasons = append(a.Reasons, "more_than_two_enemies")
	}
	if a.ThreatWeightedHP > 450 {
		a.Reasons = append(a.Reasons, "high_enemy_health_or_pressure")
	}
	if special {
		a.Reasons = append(a.Reasons, "heavy_special_or_underwater_enemy")
	}
	if v.Health < 80 {
		a.Reasons = append(a.Reasons, "low_player_health")
	}
	if v.Skill > 1 {
		a.Reasons = append(a.Reasons, "high_game_skill")
	}
	if a.NearestEnemyDistance < 128 {
		a.Reasons = append(a.Reasons, "close_initial_pressure")
	}
	if a.FarthestEnemyDistance > 640 {
		a.Reasons = append(a.Reasons, "long_initial_range")
	}
	if a.CrossfireDegrees > 100 {
		a.Reasons = append(a.Reasons, "wide_crossfire")
	}
	if a.ExpectedNonExplosiveDamage < 1.25*a.ThreatWeightedHP && a.ThreatWeightedHP > 80 {
		a.Reasons = append(a.Reasons, "insufficient_nonexplosive_reserve")
	}
	if a.ExpectedNonExplosiveSeconds > math.Min(12, a.WindowSeconds*.5) {
		a.Reasons = append(a.Reasons, "insufficient_time_reserve")
	}
	a.EligibleInitialTraining = len(a.Reasons) == 0
	switch {
	case a.EligibleInitialTraining:
		a.Tier = "beginner"
	case a.ExpectedClearSeconds > a.WindowSeconds:
		a.Tier = "resource_starved"
	case !special && len(v.Monsters) <= 3 && a.ThreatWeightedHP <= 900 && a.ExpectedClearSeconds <= a.WindowSeconds*.75:
		a.Tier = "intermediate"
	default:
		a.Tier = "hard"
	}
	return a
}

func AssessPlan(p *Plan) DifficultyReport {
	r := DifficultyReport{Version: DifficultyVersion, OutcomeIndependent: true, Counts: map[string]int{}, Cases: []EncounterDifficulty{}}
	for _, t := range p.Tasks {
		for _, v := range t.Instances {
			a := AssessEncounter(v, t.Episode.GameFrames)
			r.Cases = append(r.Cases, a)
			r.Counts[a.Tier]++
		}
	}
	return r
}

func AssessPlanFile(path string) (DifficultyReport, error) {
	var p Plan
	if err := read(path, &p); err != nil {
		return DifficultyReport{}, err
	}
	return AssessPlan(&p), nil
}

func FilterBeginnerPlan(p *Plan) (DifficultyReport, error) {
	r := AssessPlan(p)
	var selected []Task
	for _, t := range p.Tasks {
		keep := t
		keep.Seeds = nil
		keep.Instances = nil
		for _, v := range t.Instances {
			if AssessEncounter(v, t.Episode.GameFrames).EligibleInitialTraining {
				keep.Seeds = append(keep.Seeds, v.EngineSeed)
				keep.Instances = append(keep.Instances, v)
			}
		}
		if len(keep.Seeds) > 0 {
			selected = append(selected, keep)
		}
	}
	if len(selected) == 0 {
		return r, fmt.Errorf("no beginner encounters meet %s; do not relax silently", DifficultyVersion)
	}
	p.Tasks = selected
	p.DifficultyPolicy = BeginnerDifficultyPolicy
	return r, nil
}
