package trainingepisodes

import (
	"path/filepath"
	"reflect"
	"testing"
)

func TestCampaignGroupRejectsInvalidComposition(t *testing.T) {
	for _, classes := range [][]string{nil, {"monster_parasite"}, {"monster_parasite", "monster_parasite"}, {"monster_parasite", "monster_flyer"}} {
		if _, err := CampaignGroupRecipes("unused", 1, []string{"base3"}, classes); err == nil {
			t.Fatalf("invalid composition accepted: %v", classes)
		}
	}
}

func TestCampaignGroupCanonicalAndGeometryChecked(t *testing.T) {
	root, _ := filepath.Abs("../..")
	world, err := loadGenerationWorld(root, "base3")
	if err != nil {
		t.Skip(err)
	}
	a, err := CampaignGroupRecipes(root, 1, []string{"base3"}, []string{"monster_gunner", "monster_parasite", "monster_infantry"})
	if err != nil {
		t.Fatal(err)
	}
	b, err := CampaignGroupRecipes(root, 1, []string{"base3"}, []string{"monster_infantry", "monster_gunner", "monster_parasite"})
	if err != nil || !reflect.DeepEqual(a, b) {
		t.Fatalf("class order changed registered conditions: %v", err)
	}
	for _, split := range splitNames {
		instance, err := generate(a[0], split, a[0].Splits[split].Start, &world)
		if err != nil {
			t.Fatal(err)
		}
		if len(instance.Monsters) != 3 || checkStart(instance, &world) != "" {
			t.Fatal("invalid group start")
		}
		instance.Monsters[1].Position = instance.Monsters[0].Position
		if checkStart(instance, &world) == "" {
			t.Fatal("overlapping group members accepted")
		}
	}
}

func TestCampaignGroupLoadoutsHaveDisjointSeeds(t *testing.T) {
	root, _ := filepath.Abs("../..")
	if _, err := loadGenerationWorld(root, "base3"); err != nil {
		t.Skip(err)
	}
	classes := []string{"monster_parasite", "monster_infantry", "monster_gunner"}
	var ranges []Seeds
	for _, loadout := range []string{"blaster", "machinegun", "weapons", "weapons-ssg"} {
		episodes, err := CampaignGroupRecipesForLoadout(root, 1, []string{"base3"}, classes, loadout)
		if err != nil {
			t.Fatal(err)
		}
		ep := episodes[0]
		if ep.Recipe.Loadout != loadout {
			t.Fatal("incorrect inventory")
		}
		for _, split := range splitNames {
			current := ep.Splits[split]
			for _, old := range ranges {
				if current.Start <= old.Start+old.Count-1 && old.Start <= current.Start+current.Count-1 {
					t.Fatal("loadout/split seed overlap")
				}
			}
			ranges = append(ranges, current)
		}
	}
	if _, err := CampaignGroupRecipesForLoadout(root, 1, []string{"base3"}, classes, "rocketlauncher"); err == nil {
		t.Fatal("unsupported weapon accepted")
	}
}
