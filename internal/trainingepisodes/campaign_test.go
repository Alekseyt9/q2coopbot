package trainingepisodes

import (
	"path/filepath"
	"q2coopbot/internal/quake"
	"testing"
)

func TestCampaignMapSelectionRejectsInvalidAndDuplicateMaps(t *testing.T) {
	for _, maps := range [][]string{nil, {"../base1"}, {"base1", "base1"}, {"missing_map"}} {
		if _, err := CampaignRecipesForMaps("unused", 1, maps); err == nil {
			t.Fatalf("invalid map selection accepted: %v", maps)
		}
	}
}

func TestCampaignMapSubsetPreservesSeedRanges(t *testing.T) {
	root, _ := filepath.Abs("../..")
	if _, err := loadGenerationWorld(root, "base3"); err != nil {
		t.Skip(err)
	}
	selected, err := CampaignRecipesForMaps(root, 1, []string{"base3", "base1"})
	if err != nil {
		t.Fatal(err)
	}
	legacy, err := CampaignRecipes(root, 1)
	if err != nil {
		t.Fatal(err)
	}
	for _, split := range splitNames {
		if selected[1].Splits[split] != legacy[0].Splits[split] {
			t.Fatalf("subset changed old %s seeds", split)
		}
		if selected[0].Splits[split].Start < 2200000 {
			t.Fatal("base3 reused base1/base2 seed range")
		}
	}
}

func TestCampaignSitesUseDistinctOriginalGeometry(t *testing.T) {
	root, _ := filepath.Abs("../..")
	w, err := loadGenerationWorld(root, "base1")
	if err != nil {
		t.Skip(err)
	}
	_ = w
	b2, _ := loadGenerationWorld(root, "base2")
	bad := Instance{Map: "base2", Player: quake.Vec3{-96, -96, 8.125}, Monsters: []GeneratedMonster{{Class: "monster_infantry", Position: quake.Vec3{96, -96, 8.125}}}}
	if reason := checkStart(bad, &b2); reason != "inline_brush_start_overlap" {
		t.Fatalf("inline wall accepted: %s", reason)
	}
	episodes, err := CampaignRecipes(root, 4)
	if err != nil {
		t.Fatal(err)
	}
	if len(episodes) != 8 {
		t.Fatal("missing map sites")
	}
	for i, ep := range episodes {
		if ep.Generator.Site == nil {
			t.Fatal("missing source")
		}
		if ep.Map != ep.Generator.Site.Map {
			t.Fatal("wrong geometry map")
		}
		for _, other := range episodes[:i] {
			if other.Map == ep.Map && quake.Horizontal(other.Generator.Site.SourceOrigin, ep.Generator.Site.SourceOrigin) < 512 {
				t.Fatal("same arena reused")
			}
		}
		world, err := loadGenerationWorld(root, ep.Map)
		if err != nil {
			t.Fatal(err)
		}
		for _, split := range splitNames {
			if _, err := generate(ep, split, ep.Splits[split].Start, &world); err != nil {
				t.Fatal(err)
			}
		}
		world.BSPSHA256 = "changed"
		if _, err := generate(ep, "train", ep.Splits["train"].Start, &world); err == nil {
			t.Fatal("changed geometry accepted")
		}
	}
}
