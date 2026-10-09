package aimquery

import (
	"fmt"
	"math"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

const TargetedVersion = "observed_target_aim_query_v1"

// These annotations are offline training queries, never live aim assistance.
// Constant visible velocity and eye origin approximate projectile interception;
// muzzle offset, shot delay and acceleration still require native validation.
type TargetedLabel struct {
	Version     string  `json:"version"`
	Slot        int     `json:"slot"`
	Entity      int     `json:"entity"`
	Track       int     `json:"track"`
	YawDelta    float64 `json:"yaw_delta_degrees"`
	PitchDelta  float64 `json:"pitch_delta_degrees"`
	LeadKnown   bool    `json:"lead_known"`
	LeadSeconds float64 `json:"lead_seconds"`
	RecoilKnown bool    `json:"recoil_known"`
}

func Targeted(o policy.Observation) ([]TargetedLabel, error) {
	if o.Version != policy.ObservationVersion || o.Health <= 0 || o.AgeMS < 0 || o.AgeMS > 300 {
		return nil, fmt.Errorf("invalid target query observation")
	}
	labels := []TargetedLabel{}
	for slot, e := range policy.TargetEnemies(o) {
		_, known, err := policy.ObservedAimDirection(o, e)
		if err != nil {
			return nil, err
		}
		if !known || e.ID <= 0 {
			continue
		}
		point := (quake.Object{Origin: e.Relative, Solid: *e.Solid}).AimPoint()
		if o.Ducked {
			point[2] += 2
		} else {
			point[2] -= 22
		}
		label := TargetedLabel{Version: TargetedVersion, Slot: slot + 1, Entity: e.ID}
		if e.Track != nil {
			label.Track = *e.Track
		}
		// Blaster direction excludes camera kick. Machinegun uses fresh weapon
		// kick, which differs from observed camera damage/bob/kick composition.
		if o.Weapon == "Blaster" || o.Weapon == "models/weapons/v_blast/tris.md2" {
			label.RecoilKnown = true
			if e.Velocity != nil {
				v := *e.Velocity
				a := -1000000.
				b, c := 0., 0.
				for i := range v {
					if math.IsNaN(v[i]) || math.IsInf(v[i], 0) {
						return nil, fmt.Errorf("nonfinite target velocity")
					}
					a += v[i] * v[i]
					b += 2 * point[i] * v[i]
					c += point[i] * point[i]
				}
				t := math.Inf(1)
				if math.Abs(a) < 1e-9 {
					if math.Abs(b) > 1e-9 && -c/b > 0 {
						t = -c / b
					}
				} else if d := b*b - 4*a*c; d >= 0 {
					for _, r := range []float64{(-b - math.Sqrt(d)) / (2 * a), (-b + math.Sqrt(d)) / (2 * a)} {
						if r > 0 && r < t {
							t = r
						}
					}
				}
				if t <= 2 {
					for i := range point {
						point[i] += v[i] * t
					}
					label.LeadKnown = true
					label.LeadSeconds = t
				}
			}
		}
		yaw := math.Atan2(point[1], point[0])*180/math.Pi - float64(o.ViewAngles[1])*360/65536
		pitch := -math.Atan2(point[2], math.Hypot(point[0], point[1]))*180/math.Pi - float64(o.ViewAngles[0])*360/65536
		label.YawDelta = math.Remainder(yaw, 360)
		label.PitchDelta = math.Remainder(pitch, 360)
		labels = append(labels, label)
	}
	return labels, nil
}
