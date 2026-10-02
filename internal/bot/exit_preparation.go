package bot

import (
	"strings"

	"q2coopbot/internal/quake"
)

const exitPreparationFrames = 200

var ammoReserves = map[string]int{"Shells": 10, "Bullets": 40, "Cells": 40, "Rockets": 5, "Slugs": 5, "Grenades": 5}

type ExitPreparation struct {
	Map            string         `json:"map"`
	State          string         `json:"state"`
	SpentFrames    int            `json:"spent_frames"`
	BudgetFrames   int            `json:"budget_frames"`
	HealthTarget   int            `json:"health_target"`
	ArmorTarget    int            `json:"armor_target"`
	MissingHealth  int            `json:"missing_health"`
	MissingArmor   int            `json:"missing_armor"`
	MissingAmmo    map[string]int `json:"missing_ammo,omitempty"`
	MissingWeapon  bool           `json:"missing_weapon"`
	InventoryKnown bool           `json:"inventory_known"`
	lastFrame      int
}

func (p *Planner) nearCampaignExit(s quake.Snapshot) bool {
	return p.Campaign && s.Teammate == nil && s.Health > 0 && p.World.Campaign != nil && p.World.Campaign.Exit != nil && quake.Horizontal(s.Self, p.World.Campaign.Exit.Center) < 768
}

func (p *Planner) updateExitPreparation(s quake.Snapshot) {
	if p.exitPreparation == nil {
		if p.testSetupHold || !s.OnGround || !p.nearCampaignExit(s) {
			return
		}
		p.exitPreparation = &ExitPreparation{Map: s.Map, lastFrame: s.Frame}
	}
	d := p.exitPreparation
	if d.Map != s.Map {
		return
	}
	// A single per-map clock also survives leaving the exit neighbourhood.
	d.SpentFrames = min(exitPreparationFrames, d.SpentFrames+max(0, s.Frame-d.lastFrame))
	d.lastFrame = s.Frame
	d.BudgetFrames, d.HealthTarget, d.ArmorTarget = exitPreparationFrames, 75, 25
	d.MissingHealth, d.MissingArmor = max(0, 75-int(s.Health)), max(0, 25-int(s.Armor))
	d.InventoryKnown = s.InventoryKnown && s.InventoryAgeFrames <= 20
	d.MissingAmmo, d.MissingWeapon = map[string]int{}, true
	if d.InventoryKnown {
		for class, spec := range pickupSpecs {
			if !strings.HasPrefix(class, "weapon_") || inventoryCount(s, spec.name) == 0 {
				continue
			}
			d.MissingWeapon = false
			if missing := ammoReserves[spec.ammo] - inventoryCount(s, spec.ammo); missing > 0 {
				d.MissingAmmo[spec.ammo] = missing
			}
		}
	}
	d.State = "collecting"
	if d.SpentFrames >= exitPreparationFrames {
		d.State = "budget_exhausted"
	} else if d.InventoryKnown && d.MissingHealth == 0 && d.MissingArmor == 0 && len(d.MissingAmmo) == 0 && !d.MissingWeapon {
		d.State = "ready"
	}
	p.World.Campaign.Preparation = d
}

func (p *Planner) exitPreparationDone() bool {
	return p.exitPreparation != nil && (p.exitPreparation.State == "ready" || p.exitPreparation.SpentFrames >= exitPreparationFrames)
}

func (p *Planner) exitHealthAllowed(s quake.Snapshot, item quake.Object, fromMemory bool) bool {
	return s.Health < 45 || p.exitPreparation == nil || !p.nearCampaignExit(s) || p.resourceDetourAllowed(s, item, healthStand(item.Origin), fromMemory)
}

func (p *Planner) exitPickupAllowed(s quake.Snapshot, item quake.Object, at quake.Vec3, cost float64, fromMemory bool) bool {
	if !p.nearCampaignExit(s) || p.exitPreparation == nil {
		return true
	}
	// Touching a visible useful item on the way out costs no lengthy detour.
	if !fromMemory && quake.Distance(s.Self, at) <= 48 && cost <= 64 {
		return true
	}
	if p.exitPreparationDone() {
		return false
	}
	if strings.HasPrefix(item.Class, "ammo_") {
		return usefulRememberedPickup(s, item.Class)
	}
	if strings.HasPrefix(item.Class, "item_armor_") {
		return s.Armor < 25
	}
	return true
}

func (p *Planner) exitPickupRank(class string, rank int) int {
	if d := p.exitPreparation; d != nil && d.State == "collecting" {
		if strings.HasPrefix(class, "weapon_") && d.MissingWeapon {
			return 6
		}
		if spec := pickupSpecs[class]; strings.HasPrefix(class, "ammo_") && d.MissingAmmo[spec.name] > 0 {
			return 4
		}
		if strings.HasPrefix(class, "item_armor_") && d.MissingArmor > 0 {
			return 3
		}
	}
	return rank
}
