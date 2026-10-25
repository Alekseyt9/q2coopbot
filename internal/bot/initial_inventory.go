package bot

import (
	"fmt"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"strconv"
	"strings"
)

func initialInventoryNames() []string {
	return append(policy.WeaponChoices()[1:], "Bullets", "Shells", "Cells", "Rockets", "Slugs")
}

func initialInventoryCommand(items []quake.InventoryItem, weapon string) (string, error) {
	names := initialInventoryNames()
	if len(items) != len(names) {
		return "", fmt.Errorf("exact initial inventory needs16unique stock items")
	}
	allowed := map[string]int{}
	for _, name := range names {
		allowed[name] = 1
	}
	for name, limit := range map[string]int{"Bullets": 200, "Shells": 100, "Cells": 200, "Grenades": 50, "Rockets": 50, "Slugs": 50} {
		allowed[name] = limit
	}
	counts := map[string]int{}
	for _, item := range items {
		limit, ok := allowed[item.Name]
		if !ok || item.Count < 0 || item.Count > limit {
			return "", fmt.Errorf("invalid initial inventory item %q", item.Name)
		}
		if _, exists := counts[item.Name]; exists {
			return "", fmt.Errorf("duplicate initial item %q", item.Name)
		}
		counts[item.Name] = item.Count
	}
	if counts["Blaster"] != 1 {
		return "", fmt.Errorf("stock fallback Blaster required")
	}
	age := 0
	mask, err := policy.WeaponAvailability(policy.Observation{Inventory: &items, InventoryAgeFrames: &age})
	if err != nil {
		return "", err
	}
	index := 0
	for i, name := range policy.WeaponChoices() {
		if i > 0 && name == weapon && mask[i] {
			index = i
		}
	}
	if index == 0 {
		return "", fmt.Errorf("unavailable initial weapon %q", weapon)
	}
	values := make([]string, len(names))
	for i, name := range names {
		values[i] = strconv.Itoa(counts[name])
	}
	return "test_inventory_v1 " + strings.Join(values, ",") + " " + strconv.Itoa(index), nil
}

func initialInventoryReady(items []quake.InventoryItem, weapon string, s quake.Snapshot) bool {
	if !s.InventoryKnown || s.InventoryAgeFrames < 0 || s.InventoryAgeFrames > 2 {
		return false
	}
	models := []string{"v_blast", "v_shotg", "v_shotg2", "v_machn", "v_chain", "v_launch", "v_rocket", "v_hyperb", "v_rail", "v_bfg", "v_handgr"}
	matched := false
	for i, name := range policy.WeaponChoices()[1:] {
		if name == weapon && (s.Weapon == weapon || s.Weapon == "models/weapons/"+models[i]+"/tris.md2") {
			matched = true
		}
	}
	if !matched {
		return false
	}
	for _, item := range items {
		if inventoryCount(s, item.Name) != item.Count {
			return false
		}
	}
	return true
}
