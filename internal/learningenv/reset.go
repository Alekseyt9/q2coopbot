package learningenv

import (
	"fmt"
	"math"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

const ResetVersion = "observed_fixture_reset_v1"

// This verifies the exposed starting fields, not inventory/RNG/monster AI or
// complete server reset equivalence. The expectation is an offline fixture.
type ResetExpectation struct {
	Version       string     `json:"version"`
	Map           string     `json:"map"`
	Position      quake.Vec3 `json:"position"`
	Health        int16      `json:"health"`
	Armor         int16      `json:"armor"`
	Weapon        string     `json:"weapon"`
	Ammo          int16      `json:"ammo"`
	EnemyClass    string     `json:"enemy_class"`
	EnemyPosition quake.Vec3 `json:"enemy_position"`
}

type ResetProof struct {
	Version                  string           `json:"version"`
	ObservedFieldsConfirmed  bool             `json:"observed_fields_confirmed"`
	FullServerResetConfirmed bool             `json:"full_server_reset_confirmed"`
	Expectation              ResetExpectation `json:"expectation"`
	Reason                   string           `json:"reason,omitempty"`
	Unverified               []string         `json:"unverified"`
}

func VerifyReset(o policy.Observation, expected ResetExpectation) ResetProof {
	r := ResetProof{Version: ResetVersion, Expectation: expected, Unverified: []string{"inventory", "server_rng", "monster_ai_state", "entity_generation", "complete_world_reset"}}
	if expected.Version != ResetVersion || expected.Map == "" || expected.Health <= 0 || expected.EnemyClass == "" || expected.Weapon != "Shotgun" {
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
	case o.Weapon != "Shotgun" && o.Weapon != "models/weapons/v_shotg/tris.md2":
		r.Reason = "reset_weapon_mismatch"
	}
	if r.Reason != "" {
		return r
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
