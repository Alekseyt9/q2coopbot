package learningenv

import (
	"fmt"
	"math"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

const ResetVersion = "observed_fixture_reset_v1"

// This verifies exposed starting fields and optionally listed inventory, not RNG/monster AI or
// complete server reset equivalence. The expectation is an offline fixture.
type ResetExpectation struct {
	AllowUnobservedEnemy bool                  `json:"allow_unobserved_enemy,omitempty"`
	Seed                 *int                  `json:"seed,omitempty"`
	Version              string                `json:"version"`
	Map                  string                `json:"map"`
	Position             quake.Vec3            `json:"position"`
	Health               int16                 `json:"health"`
	Armor                int16                 `json:"armor"`
	Weapon               string                `json:"weapon"`
	Ammo                 int16                 `json:"ammo"`
	EnemyClass           string                `json:"enemy_class"`
	EnemyPosition        quake.Vec3            `json:"enemy_position"`
	Inventory            []quake.InventoryItem `json:"inventory,omitempty"`
}

type ResetProof struct {
	NativeBarrier            *CombatRelease   `json:"native_barrier,omitempty"`
	Version                  string           `json:"version"`
	ObservedFieldsConfirmed  bool             `json:"observed_fields_confirmed"`
	FullServerResetConfirmed bool             `json:"full_server_reset_confirmed"`
	Expectation              ResetExpectation `json:"expectation"`
	Reason                   string           `json:"reason,omitempty"`
	Unverified               []string         `json:"unverified"`
}

func VerifyReset(o policy.Observation, expected ResetExpectation) ResetProof {
	r := ResetProof{Version: ResetVersion, Expectation: expected, Unverified: []string{"inventory", "server_rng", "monster_ai_state", "entity_generation", "complete_world_reset"}}
	if expected.AllowUnobservedEnemy {
		r.Reason = "unobserved_enemy_reset_requires_paired_proof"
		return r
	}
	if expected.Version != ResetVersion || expected.Map == "" || expected.Health <= 0 || expected.EnemyClass == "" || expected.Weapon != "Shotgun" && expected.Weapon != "Blaster" && expected.Weapon != "Machinegun" {
		r.Reason = "invalid_or_unsupported_reset_expectation"
		return r
	}
	for _, point := range []quake.Vec3{expected.Position, expected.EnemyPosition, o.Position} {
		for _, v := range point {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				r.Reason = "nonfinite_reset_position"
				return r
			}
		}
	}
	id := o.Identity
	switch {
	case o.Version != policy.ObservationVersion || id.Map != expected.Map || id.Connection < 1 || id.Actor < 1 || id.Frame < 1 || id.Life != 1:
		r.Reason = "reset_identity_mismatch"
	case o.AgeMS < 0 || o.AgeMS > 300:
		r.Reason = "stale_reset_observation"
	case o.Health != expected.Health || o.Armor != expected.Armor || o.Ammo != expected.Ammo:
		r.Reason = "reset_resources_mismatch"
	case !o.OnGround || o.Ducked || !near(o.Position, expected.Position):
		r.Reason = "reset_pose_mismatch"
	case o.Weapon != expected.Weapon && !(expected.Weapon == "Shotgun" && o.Weapon == "models/weapons/v_shotg/tris.md2") && !(expected.Weapon == "Machinegun" && o.Weapon == "models/weapons/v_machn/tris.md2"):
		r.Reason = "reset_weapon_mismatch"
	}
	if r.Reason != "" {
		return r
	}
	if len(expected.Inventory) > 0 {
		if o.Inventory == nil || o.InventoryAgeFrames == nil || *o.InventoryAgeFrames < 0 || *o.InventoryAgeFrames > 2 {
			r.Reason = "reset_inventory_unknown_or_stale"
			return r
		}
		counts := map[string]int{}
		for _, item := range *o.Inventory {
			counts[item.Name] = item.Count
		}
		for _, item := range expected.Inventory {
			count, known := counts[item.Name]
			if item.Name == "" || item.Count < 0 || !known || count != item.Count {
				r.Reason = "reset_inventory_mismatch"
				return r
			}
		}
		r.Unverified = []string{"unlisted_inventory", "server_rng", "monster_ai_state", "entity_generation", "complete_world_reset"}
	}
	found := false
	for _, e := range o.Enemies {
		pos := quake.Vec3{}
		for axis := range pos {
			pos[axis] = o.Position[axis] + e.Relative[axis]
		}
		if e.Class == expected.EnemyClass && e.ClearShot != nil && *e.ClearShot && near(pos, expected.EnemyPosition) {
			found = true
		}
	}
	if !found {
		r.Reason = "reset_observed_enemy_mismatch"
		return r
	}
	r.ObservedFieldsConfirmed = true
	return r
}

// A real peer can have its enemy occluded at reset. Only paired export may
// opt into a participant-resource proof; enemy/world reset remains unverified.
func VerifyPairedParticipantReset(o policy.Observation, expected ResetExpectation) ResetProof {
	copy := expected
	copy.AllowUnobservedEnemy = false
	r := VerifyReset(o, copy)
	r.Expectation = expected
	if expected.AllowUnobservedEnemy && r.Reason == "reset_observed_enemy_mismatch" {
		r.Reason = ""
		r.ObservedFieldsConfirmed = true
		r.Unverified = append(r.Unverified, "enemy_presence_and_pose_not_observed")
	}
	return r
}

func near(a, b quake.Vec3) bool {
	for i := range a {
		if math.IsNaN(a[i]) || math.IsInf(a[i], 0) || math.Abs(a[i]-b[i]) > 1 {
			return false
		}
	}
	return true
}

func (r ResetProof) Error() error {
	if r.ObservedFieldsConfirmed {
		return nil
	}
	return fmt.Errorf("observed fixture reset failed: %s", r.Reason)
}
