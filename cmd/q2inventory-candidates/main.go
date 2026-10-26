// Inventory-only preflight; no BSP, native world or neural forward execution.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"q2coopbot/internal/trainingepisodes"
)

type candidate struct {
	Episode   string                              `json:"episode_id"`
	Map       string                              `json:"map"`
	Seed      int                                 `json:"seed"`
	Offset    int                                 `json:"offset"`
	Inventory trainingepisodes.GeneratedInventory `json:"inventory"`
}

func selectCandidates(episodes []trainingepisodes.Episode, ids []string, split string, offset, count int, weapon, ammo string, minimum int) ([]candidate, error) {
	if count < 1 || count > 512 || offset < 0 || minimum < 0 {
		return nil, fmt.Errorf("invalid candidate range")
	}
	wanted := map[string]bool{}
	for _, id := range ids {
		if id == "" || wanted[id] {
			return nil, fmt.Errorf("empty/duplicate episode")
		}
		wanted[id] = true
	}
	var result []candidate
	for _, ep := range episodes {
		if !wanted[ep.ID] {
			continue
		}
		delete(wanted, ep.ID)
		s, ok := ep.Splits[split]
		if !ok || count > s.Count || offset > s.Count-count || ep.Status != "runnable" || ep.Generator == nil || ep.Generator.Inventory == nil {
			return nil, fmt.Errorf("invalid inventory cohort for %s", ep.ID)
		}
		for i := 0; i < count; i++ {
			seed := s.Start + offset + i
			inv, err := trainingepisodes.SampleInitialInventory(*ep.Generator.Inventory, seed)
			if err != nil {
				return nil, err
			}
			if weapon != "" && inv.InitialWeapon != weapon {
				continue
			}
			if ammo != "" {
				amount := -1
				for _, item := range inv.Items {
					if item.Name == ammo {
						amount = item.Count
					}
				}
				if amount < minimum {
					continue
				}
			}
			result = append(result, candidate{Episode: ep.ID, Map: ep.Map, Seed: seed, Offset: offset + i, Inventory: inv})
		}
	}
	if len(wanted) > 0 {
		return nil, fmt.Errorf("unknown episode selection")
	}
	return result, nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	registry := flag.String("registry", "scripts/scenarios/combat-training/index.json", "registry")
	ids := flag.String("episodes", "", "comma-separated IDs")
	split := flag.String("split", "validation", "registered split")
	offset := flag.Int("seed-offset", 0, "first seed offset")
	count := flag.Int("count", 32, "inventory draws per descriptor")
	weapon := flag.String("initial-weapon", "", "optional observed initial weapon filter")
	ammo := flag.String("ammo", "", "optional ammo item")
	minimum := flag.Int("minimum-ammo", 0, "minimum count")
	out := flag.String("out", "", "fresh metadata JSON")
	flag.Parse()
	r, err := trainingepisodes.Load(*registry)
	if err != nil {
		return err
	}
	c, err := selectCandidates(r.Episodes, strings.Split(*ids, ","), *split, *offset, *count, *weapon, *ammo, *minimum)
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(map[string]any{"version": "combat_inventory_candidates_v1", "registry_path": r.Path, "registry_sha256": r.SHA256, "split": *split, "candidates": c, "geometry_verified": false, "native_games": 0}, "", "  ")
	if err != nil {
		return err
	}
	f, err := os.OpenFile(*out, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0666)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(data)
	return err
}
