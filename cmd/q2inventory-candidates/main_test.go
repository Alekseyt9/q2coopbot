package main

import (
	"q2coopbot/internal/trainingepisodes"
	"reflect"
	"testing"
)

func TestInventoryCandidatesMatchIndependentSampler(t *testing.T) {
	d := trainingepisodes.InventoryDistribution{Version: 1, WeaponPool: []string{"Machinegun"}, MinWeapons: 1, MaxWeapons: 1,
		AmmoCounts: map[string][]int{"Bullets": {0, 5, 20}, "Shells": {0}, "Cells": {0}, "Grenades": {0}, "Rockets": {0}, "Slugs": {0}}}
	ep := trainingepisodes.Episode{ID: "fixture", Map: "mine2", Status: "runnable", Splits: map[string]trainingepisodes.Seeds{"validation": {Start: 1000, Count: 512}}, Generator: &trainingepisodes.Generator{Inventory: &d}}
	r, err := selectCandidates([]trainingepisodes.Episode{ep}, []string{ep.ID}, "validation", 320, 32, "Machinegun", "Bullets", 5)
	if err != nil || len(r) == 0 {
		t.Fatal(err, len(r))
	}
	for _, c := range r {
		expected, err := trainingepisodes.SampleInitialInventory(d, c.Seed)
		if err != nil || !reflect.DeepEqual(c.Inventory, expected) || c.Seed < 1320 || c.Seed >= 1352 || c.Inventory.InitialWeapon != "Machinegun" {
			t.Fatal(c, err)
		}
	}
	for _, ids := range [][]string{{"unknown"}, {ep.ID, ep.ID}} {
		if _, err := selectCandidates([]trainingepisodes.Episode{ep}, ids, "validation", 320, 32, "", "", 0); err == nil {
			t.Fatal("invalid selection accepted")
		}
	}
	if _, err := selectCandidates([]trainingepisodes.Episode{ep}, []string{ep.ID}, "validation", 500, 32, "", "", 0); err == nil {
		t.Fatal("range escaped split")
	}
}
