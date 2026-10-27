package trainingepisodes

import (
	"q2coopbot/internal/quake"
	"reflect"
	"testing"
)

func difficultyFixture(classes []string, items map[string]int) Instance {
	v := Instance{EpisodeID: "difficulty-fixture", Map: "jail4", Health: 100, Skill: 1, Inventory: &GeneratedInventory{}}
	for key, value := range items {
		v.Inventory.Items = append(v.Inventory.Items, InventoryCount{Name: key, Count: value})
	}
	for i, class := range classes {
		v.Monsters = append(v.Monsters, GeneratedMonster{Class: class, Position: quake.Vec3{256, float64(i * 32), 0}})
	}
	return v
}

func TestEncounterDifficultyJail4OneSharedGrenade(t *testing.T) {
	v := difficultyFixture([]string{"monster_gunner", "monster_gunner", "monster_tank"}, map[string]int{
		"Blaster": 1, "Grenade Launcher": 1, "Grenades": 1, "Cells": 49, "Rockets": 3})
	a := AssessEncounter(v, 300)
	if a.EligibleInitialTraining || a.Tier != "resource_starved" || a.StockMonsterHP != 1100 || a.ExpectedClearSeconds <= a.WindowSeconds {
		t.Fatalf("hopeless beginner assignment accepted: %+v", a)
	}
	if a.ExpectedFiniteDamage > 125 || a.ExpectedNonExplosiveDamage != 0 {
		t.Fatal("shared grenade or unowned ammo counted twice", a)
	}
}

func TestEncounterDifficultyUsableAmmoAndHealth(t *testing.T) {
	v := difficultyFixture([]string{"monster_infantry"}, map[string]int{"Blaster": 1, "Machinegun": 1, "Bullets": 50})
	if !AssessEncounter(v, 300).EligibleInitialTraining {
		t.Fatal("ordinary well-equipped infantry lesson rejected")
	}
	for _, change := range []func(*Instance){
		func(v *Instance) { v.Health = 30 },
		func(v *Instance) { v.Skill = 3 },
		func(v *Instance) { v.Monsters[0].Position = quake.Vec3{32, 0, 0} },
	} {
		bad := v
		bad.Monsters = append([]GeneratedMonster(nil), v.Monsters...)
		change(&bad)
		if AssessEncounter(bad, 300).EligibleInitialTraining {
			t.Fatal("pressure/health/skill did not affect initial eligibility")
		}
	}
	starved := difficultyFixture([]string{"monster_infantry"}, map[string]int{"Blaster": 1, "Machinegun": 1, "Cells": 200})
	if a := AssessEncounter(starved, 300); a.EligibleInitialTraining || a.ExpectedFiniteDamage != 0 {
		t.Fatal("unusable or unowned ammo funds a weapon", a)
	}
	bfg := difficultyFixture([]string{"monster_infantry"}, map[string]int{"Blaster": 1, "BFG10K": 1, "Cells": 49})
	if a := AssessEncounter(bfg, 300); a.ExpectedFiniteDamage != 0 {
		t.Fatal("BFG funded below 50 cells", a)
	}
}

func TestEncounterDifficultySharedPoolsAndBlasterWindow(t *testing.T) {
	only := difficultyFixture([]string{"monster_infantry"}, map[string]int{"Blaster": 1, "HyperBlaster": 1, "Cells": 100})
	both := difficultyFixture([]string{"monster_infantry"}, map[string]int{"Blaster": 1, "HyperBlaster": 1, "BFG10K": 1, "Cells": 100})
	if AssessEncounter(only, 300).ExpectedFiniteDamage != AssessEncounter(both, 300).ExpectedFiniteDamage {
		t.Fatal("shared cells double counted")
	}
	weak := difficultyFixture([]string{"monster_soldier_light", "monster_soldier_light"}, map[string]int{"Blaster": 1})
	if !AssessEncounter(weak, 300).EligibleInitialTraining {
		t.Fatal("small blaster lesson rejected")
	}
	heavy := difficultyFixture([]string{"monster_tank"}, map[string]int{"Blaster": 1})
	if a := AssessEncounter(heavy, 300); a.Tier != "resource_starved" || a.EligibleInitialTraining {
		t.Fatal("infinite blaster bypassed deadline", a)
	}
	heavy.Inventory = only.Inventory
	if AssessEncounter(heavy, 300).EligibleInitialTraining {
		t.Fatal("tank became beginner solely from sufficient cells")
	}
}

func TestBeginnerDifficultyFilterPreservesConditions(t *testing.T) {
	easy := difficultyFixture([]string{"monster_infantry"}, map[string]int{"Blaster": 1, "Machinegun": 1, "Bullets": 100})
	hard := difficultyFixture([]string{"monster_gunner", "monster_tank"}, map[string]int{"Blaster": 1, "Grenades": 1})
	easy.EngineSeed = 1
	hard.EngineSeed = 2
	last := easy
	last.EngineSeed = 3
	p := &Plan{Tasks: []Task{{Episode: Episode{GameFrames: 300}, Seeds: []int{1, 2, 3}, Instances: []Instance{easy, hard, last}}}}
	r, err := FilterBeginnerPlan(p)
	if err != nil || r.Counts["beginner"] != 2 || p.DifficultyPolicy != BeginnerDifficultyPolicy {
		t.Fatal("bad selection", err, r)
	}
	if !reflect.DeepEqual(p.Tasks[0].Seeds, []int{1, 3}) || !reflect.DeepEqual(p.Tasks[0].Instances, []Instance{easy, last}) {
		t.Fatal("filter changed inventory, actors or geometry")
	}
	empty := &Plan{Tasks: []Task{{Episode: Episode{GameFrames: 300}, Seeds: []int{2}, Instances: []Instance{hard}}}}
	if _, err := FilterBeginnerPlan(empty); err == nil {
		t.Fatal("empty curriculum silently relaxed")
	}
	unknown := easy
	unknown.Monsters = []GeneratedMonster{{Class: "monster_unknown"}}
	if a := AssessEncounter(unknown, 300); a.Tier != "unassessed" || a.EligibleInitialTraining {
		t.Fatal("unknown profile admitted", a)
	}
}
