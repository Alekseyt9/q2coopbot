package trainingepisodes

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math/rand"
	"strconv"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

// InventoryDistribution describes initial conditions, never a weapon policy.
// Min/Max count sampled optional slots; Blaster is the ordinary stock fallback.
type InventoryDistribution struct {
	Version    int              `json:"version"`
	WeaponPool []string         `json:"weapon_pool"`
	MinWeapons int              `json:"min_weapons"`
	MaxWeapons int              `json:"max_weapons"`
	AmmoCounts map[string][]int `json:"ammo_counts"`
}

type InventoryCount struct {
	Name  string `json:"name"`
	Count int    `json:"count"`
}

type GeneratedInventory struct {
	Version          int              `json:"version"`
	Seed             int              `json:"seed"`
	RequestedWeapons []string         `json:"requested_weapons"`
	Items            []InventoryCount `json:"items"`
	InitialWeapon    string           `json:"initial_weapon"`
}

var inventoryAmmoOrder = []string{"Bullets", "Shells", "Cells", "Grenades", "Rockets", "Slugs"}
var inventoryAmmoLimits = map[string]int{"Bullets": 200, "Shells": 100, "Cells": 200, "Grenades": 50, "Rockets": 50, "Slugs": 50}

func (d InventoryDistribution) validate() error {
	if d.Version != 1 || d.MinWeapons < 0 || d.MaxWeapons < d.MinWeapons || d.MaxWeapons > len(d.WeaponPool) || len(d.AmmoCounts) != 6 {
		return fmt.Errorf("invalid random inventory distribution")
	}
	known := map[string]bool{}
	for _, name := range policy.WeaponChoices()[2:] {
		known[name] = true
	}
	seen := map[string]bool{}
	for _, name := range d.WeaponPool {
		if !known[name] || seen[name] {
			return fmt.Errorf("unsupported or duplicate inventory weapon %q", name)
		}
		seen[name] = true
	}
	for _, name := range inventoryAmmoOrder {
		counts := d.AmmoCounts[name]
		if len(counts) == 0 {
			return fmt.Errorf("missing ammo distribution %s", name)
		}
		for _, n := range counts {
			if n < 0 || n > inventoryAmmoLimits[name] {
				return fmt.Errorf("invalid ammo count %s=%d", name, n)
			}
		}
	}
	return nil
}

// SampleInitialInventory has a separate deterministic random stream: adding
// equipment randomization cannot consume the existing map-position RNG.
func SampleInitialInventory(d InventoryDistribution, seed int) (GeneratedInventory, error) {
	if err := d.validate(); err != nil {
		return GeneratedInventory{}, err
	}
	if seed < 0 || int64(seed) > 2147483647 {
		return GeneratedInventory{}, fmt.Errorf("invalid inventory seed")
	}
	digest := sha256.Sum256([]byte("combat-inventory-v1:" + strconv.Itoa(seed)))
	r := rand.New(rand.NewSource(int64(binary.LittleEndian.Uint64(digest[:8]))))
	allowed := map[string]bool{}
	for _, name := range d.WeaponPool {
		allowed[name] = true
	}
	var pool []string
	// Stable serialized policy order, independent of caller slice/map order.
	for _, name := range policy.WeaponChoices()[2:] {
		if allowed[name] {
			pool = append(pool, name)
		}
	}
	count := d.MinWeapons + r.Intn(d.MaxWeapons-d.MinWeapons+1)
	chosen := map[string]bool{"Blaster": true}
	for _, i := range r.Perm(len(pool))[:count] {
		chosen[pool[i]] = true
	}
	result := GeneratedInventory{Version: 1, Seed: seed}
	counts := map[string]int{"Blaster": 1}
	for _, name := range policy.WeaponChoices()[2:] {
		if chosen[name] {
			result.RequestedWeapons = append(result.RequestedWeapons, name)
			if name != "Grenades" {
				counts[name] = 1
			}
		}
	}
	for _, name := range inventoryAmmoOrder {
		values := d.AmmoCounts[name]
		n := values[r.Intn(len(values))]
		// Grenades is one native inventory item for both hand grenades and
		// launcher ammo. Positive launcher ammo also enables hand grenades;
		// an empty grenade stack cannot represent a separately owned weapon.
		if name == "Grenades" && !chosen["Grenades"] && !chosen["Grenade Launcher"] {
			n = 0
		}
		counts[name] = n
	}
	seen := map[string]bool{}
	for _, name := range append(policy.WeaponChoices()[1:], inventoryAmmoOrder...) {
		if seen[name] {
			continue
		}
		seen[name] = true
		result.Items = append(result.Items, InventoryCount{Name: name, Count: counts[name]})
	}
	var observed []quake.InventoryItem
	for _, item := range result.Items {
		observed = append(observed, quake.InventoryItem{Name: item.Name, Count: item.Count})
	}
	age := 0
	mask, err := policy.WeaponAvailability(policy.Observation{Inventory: &observed, InventoryAgeFrames: &age})
	if err != nil {
		return GeneratedInventory{}, err
	}
	var usable []string
	for i, name := range policy.WeaponChoices() {
		if i > 0 && mask[i] {
			usable = append(usable, name)
		}
	}
	if len(usable) == 0 {
		return GeneratedInventory{}, fmt.Errorf("initial inventory lacks usable Blaster")
	}
	result.InitialWeapon = usable[r.Intn(len(usable))]
	return result, nil
}
