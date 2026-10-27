package trainingepisodes

import (
	"path/filepath"
	"q2coopbot/internal/quake"
	"testing"
)

func TestBeginnerVisibilityRejectsLowerFloorFact1(t *testing.T) {
	root, _ := filepath.Abs(filepath.Join("..", ".."))
	world, err := loadGenerationWorld(root, "fact1")
	if err != nil {
		t.Skipf("local BSP required: %v", err)
	}
	v := difficultyFixture([]string{"monster_berserk", "monster_soldier_ss"}, map[string]int{
		"Blaster": 1, "Machinegun": 1, "Bullets": 100, "HyperBlaster": 1, "Cells": 50, "Railgun": 1, "Slugs": 1})
	v.Map = "fact1"
	v.Player = quake.Vec3{-555.625, 814.25, 24.125}
	v.Health = 80
	v.Monsters[0].Position = quake.Vec3{-487.375, 638.125, 24.125}
	v.Monsters[1].Position = quake.Vec3{-486.75, 768.5, -103.875}
	if !AssessEncounter(v, 300).EligibleInitialTraining {
		t.Fatal("regression no longer exercises the resource-only false positive")
	}
	a := assessVisibleEncounter(v, 300, &world)
	if a.EligibleInitialTraining || a.Tier != "geometry_blocked" || a.StartVisibleMonsters != 1 || len(a.BlockedMonsterIndices) != 1 || a.BlockedMonsterIndices[0] != 1 {
		t.Fatalf("hidden lower-floor target admitted: %+v", a)
	}
	// A solo copy of the actually visible target must remain eligible.
	v.Monsters = v.Monsters[:1]
	if a := assessVisibleEncounter(v, 300, &world); !a.EligibleInitialTraining || a.StartVisibleMonsters != 1 {
		t.Fatal("visible target rejected", a)
	}
	if assessVisibleEncounter(v, 300, nil).EligibleInitialTraining {
		t.Fatal("missing geometry silently accepted")
	}
}
