package bot

import (
	"math"
	"q2coopbot/internal/quake"
	"strings"
)

type pickupSpec struct {
	name, ammo string
	cap, rank  int
}

var pickupSpecs = map[string]pickupSpec{
	"weapon_shotgun": {"Shotgun", "Shells", 1, 3}, "weapon_supershotgun": {"Super Shotgun", "Shells", 1, 3},
	"weapon_machinegun": {"Machinegun", "Bullets", 1, 3}, "weapon_chaingun": {"Chaingun", "Bullets", 1, 3},
	"weapon_grenadelauncher": {"Grenade Launcher", "Grenades", 1, 3}, "weapon_rocketlauncher": {"Rocket Launcher", "Rockets", 1, 3},
	"weapon_hyperblaster": {"HyperBlaster", "Cells", 1, 3}, "weapon_railgun": {"Railgun", "Slugs", 1, 3}, "weapon_bfg": {"BFG10K", "Cells", 1, 3},
	"ammo_shells": {"Shells", "", 100, 2}, "ammo_bullets": {"Bullets", "", 200, 2}, "ammo_cells": {"Cells", "", 200, 2},
	"ammo_rockets": {"Rockets", "", 50, 2}, "ammo_slugs": {"Slugs", "", 50, 2}, "ammo_grenades": {"Grenades", "", 50, 2},
	"item_armor_jacket": {"Jacket Armor", "", 50, 1}, "item_armor_combat": {"Combat Armor", "", 100, 1}, "item_armor_body": {"Body Armor", "", 200, 1},
	"item_armor_shard": {"", "", 200, 1},
}

type PickupAttempt struct {
	FromMemory bool       `json:"from_memory,omitempty"`
	Entity     int        `json:"entity"`
	Class      string     `json:"class"`
	Name       string     `json:"name"`
	Target     quake.Vec3 `json:"target"`
	Started    int        `json:"started_frame"`
	Ended      int        `json:"end_frame,omitempty"`
	State      string     `json:"state"`
	Before     int        `json:"before"`
	After      int        `json:"after,omitempty"`
}
type pickupTask struct {
	attempt           PickupAttempt
	last              quake.Vec3
	progress, missing int
}

func inventoryCount(s quake.Snapshot, name string) int {
	for _, item := range s.Inventory {
		if strings.EqualFold(item.Name, name) {
			return item.Count
		}
	}
	return 0
}
func pickupCount(s quake.Snapshot, spec pickupSpec) int {
	if spec.name == "" {
		return int(s.Armor)
	}
	return inventoryCount(s, spec.name)
}
func usefulPickup(s quake.Snapshot, class string) bool {
	sp, ok := pickupSpecs[class]
	if !ok || !s.InventoryKnown || s.InventoryAgeFrames > 20 {
		return false
	}
	if pickupCount(s, sp) >= sp.cap {
		return false
	}
	if strings.HasPrefix(class, "ammo_") {
		if class == "ammo_grenades" {
			return true
		}
		for _, w := range pickupSpecs {
			if w.ammo == sp.name && inventoryCount(s, w.name) > 0 {
				return true
			}
		}
		return false
	}
	if class == "item_armor_jacket" && (inventoryCount(s, "Combat Armor") > 0 || inventoryCount(s, "Body Armor") > 0) {
		return false
	}
	if class == "item_armor_combat" && inventoryCount(s, "Body Armor") > 0 {
		return false
	}
	return true
}
func (p *Planner) finishPickup(s quake.Snapshot, state string) {
	a := p.pickup.attempt
	a.State, a.Ended, a.After = state, s.Frame, pickupCount(s, pickupSpecs[a.Class])
	p.World.Pickup = &a
	if p.pickupBanned == nil {
		p.pickupBanned = map[quake.Vec3]int{}
	}
	p.pickupBanned[a.Target] = s.Frame + 150
	p.pickup = nil
	p.pickupNext = s.Frame + 10
	p.routeKnown = false
}
func (p *Planner) pickupGoal(s quake.Snapshot) (quake.Vec3, bool) {
	p.World.ResourceYield = nil
	allowed := !p.testSetupHold && s.Health >= 45 && s.Teammate != nil && quake.Horizontal(s.Self, *s.Teammate) < 384 &&
		(p.World.Goal == "follow_teammate" || p.World.Goal == "cover_teammate") && p.elevator == nil && p.button == nil && p.jump == nil
	if p.pickup != nil {
		t := p.pickup
		if s.Health > 0 && quake.Distance(s.Self, t.attempt.Target) <= 64 && s.InventoryKnown && s.InventoryAgeFrames <= 20 && pickupCount(s, pickupSpecs[t.attempt.Class]) > t.attempt.Before {
			p.finishPickup(s, "confirmed")
			return quake.Vec3{}, false
		}
		if !allowed || !s.InventoryKnown || s.InventoryAgeFrames > 20 || s.Frame < t.attempt.Started {
			p.finishPickup(s, "interrupted")
			return quake.Vec3{}, false
		}
		visible := false
		for _, item := range s.Pickups {
			if item.ID == t.attempt.Entity && item.Class == t.attempt.Class && quake.Distance(healthStand(item.Origin), t.attempt.Target) < 8 {
				visible = true
				break
			}
		}
		if visible {
			if p.yieldPickup(s, t.attempt.Entity, t.attempt.Class, t.attempt.Target) {
				p.finishPickup(s, "yielded")
				return quake.Vec3{}, false
			}
			t.missing = 0
		} else if t.missing == 0 {
			t.missing = s.Frame
		}
		checkingMemory := false
		if r := p.resources[t.attempt.Entity]; r != nil && r.State == "unknown" && healthStand(r.Item.Origin) == t.attempt.Target {
			checkingMemory = true
			r.Attempted = true
		}
		if t.missing != 0 && s.Frame-t.missing >= 5 && !checkingMemory {
			p.finishPickup(s, "unconfirmed")
			return quake.Vec3{}, false
		}
		if quake.Distance(s.Self, t.last) > 16 {
			t.last = s.Self
			t.progress = s.Frame
		}
		if s.Frame-t.progress >= 25 || s.Frame-t.attempt.Started >= 100 {
			p.finishPickup(s, "no_progress")
			return quake.Vec3{}, false
		}
		p.World.Pickup = &t.attempt
		return t.attempt.Target, true
	}
	if !allowed || s.Frame < p.pickupNext || !s.OnGround || p.Nav == nil || !p.World.Geometry.HasCollision() {
		return quake.Vec3{}, false
	}
	best := math.Inf(1)
	var selected *pickupTask
	candidates := append([]quake.Object(nil), s.Pickups...)
	candidates = append(candidates, p.rememberedCandidates(s)...)
	for _, item := range candidates {
		if !usefulPickup(s, item.Class) {
			continue
		}
		at := healthStand(item.Origin)
		if s.Frame < p.pickupBanned[at] || quake.Distance(s.Self, at) > 256 || quake.Distance(*s.Teammate, at) > 384 {
			continue
		}
		cost, ok := p.resourceRoute(s.Self, at)
		if !ok {
			continue
		}
		if p.yieldPickup(s, item.ID, item.Class, at) {
			continue
		}
		sp := pickupSpecs[item.Class]
		score := cost / float64(sp.rank)
		if score >= best {
			continue
		}
		best = score
		selected = &pickupTask{attempt: PickupAttempt{Entity: item.ID, Class: item.Class, Name: sp.name, Target: at, Started: s.Frame, State: "approach", Before: pickupCount(s, sp)}, last: s.Self, progress: s.Frame}
	}
	if selected == nil {
		return quake.Vec3{}, false
	}
	p.pickup = selected
	if r := p.resources[selected.attempt.Entity]; r != nil && r.State == "unknown" {
		selected.attempt.FromMemory = true
		p.markResourceVisit(selected.attempt.Target)
	}
	p.World.Pickup = &selected.attempt
	p.routeKnown = false
	return selected.attempt.Target, true
}
