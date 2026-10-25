package policy

import (
	"fmt"
	"math"
	"strings"

	"q2coopbot/internal/quake"
)

const BodyFeatureVersion = "combat_features_v11"
const BodyAblatedFeatureVersion = "combat_features_v11_body_ablated"
const BodyFeatureTailWidth = 2 + dynamicBodyCapacity*20
const BodyFeatureWidth = EquippedWeaponFeatureWidth + BodyFeatureTailWidth

func equippedPrefixFeatures(version string) bool {
	return version == EquippedWeaponFeatureVersion || version == BodyFeatureVersion || version == BodyAblatedFeatureVersion ||
		version == ProjectileTypeFeatureVersion || version == ProjectileTypeNeutralFeatureVersion || version == ProjectileTypeAblatedFeatureVersion || version == MemoryRouteFeatureVersion
}

// The old v10 prefix is untouched. Neither entity IDs nor native labels enter
// the tail. Unknown protocol shape stays masked, including solid0 corpses.
func bodyFeatures(o Observation) ([]float64, error) {
	b := func(value bool) float64 {
		if value {
			return 1
		}
		return 0
	}
	result := make([]float64, BodyFeatureTailWidth)
	set := o.DynamicBodies
	if set == nil {
		return result, nil
	}
	if set.Version != DynamicBodyObservationVersion || set.Frame != o.Identity.Frame || set.Range != dynamicBodyRange || set.Capacity != dynamicBodyCapacity || len(set.Bodies) > dynamicBodyCapacity {
		return nil, fmt.Errorf("invalid dynamic body observation contract")
	}
	result[0], result[1] = 1, b(set.Truncated)
	yaw := degrees(o.ViewAngles[1]) * math.Pi / 180
	cy, sy := math.Cos(yaw), math.Sin(yaw)
	seen := map[int]bool{}
	for index, body := range set.Bodies {
		if body.ID <= 0 || body.ID == o.Identity.Actor || seen[body.ID] || body.Distance < 0 || body.Distance > dynamicBodyRange {
			return nil, fmt.Errorf("invalid observed body identity/range")
		}
		seen[body.ID] = true
		if index > 0 {
			previous := set.Bodies[index-1]
			if body.Distance < previous.Distance || body.Distance == previous.Distance && body.ID <= previous.ID {
				return nil, fmt.Errorf("unsorted dynamic bodies")
			}
		}
		if body.ProtocolBoundsKnown != (body.ProtocolMins != nil && body.ProtocolMaxs != nil) || !body.ProtocolBoundsKnown && (body.ProtocolMins != nil || body.ProtocolMaxs != nil) {
			return nil, fmt.Errorf("invalid protocol body shape mask")
		}
		if body.ProtocolBoundsKnown && (body.Solid == 0 || body.Solid == 31) {
			return nil, fmt.Errorf("unknown solid materialized as body bounds")
		}
		path := strings.ToLower(body.ModelPath)
		at := 2 + 20*index
		r := body.Relative
		values := []float64{1, (r[0]*cy + r[1]*sy) / 512, (-r[0]*sy + r[1]*cy) / 512, r[2] / 512, body.Distance / 512, b(body.ProtocolBoundsKnown)}
		if body.ProtocolBoundsKnown {
			lo, hi, ok := (quake.Object{Solid: body.Solid}).SolidBounds()
			if !ok || *body.ProtocolMins != lo || *body.ProtocolMaxs != hi {
				return nil, fmt.Errorf("body bounds differ from client protocol solid")
			}
			for _, value := range lo {
				values = append(values, value/64)
			}
			for _, value := range hi {
				values = append(values, value/64)
			}
		} else {
			values = append(values, 0, 0, 0, 0, 0, 0)
		}
		bodyYaw := body.Angles[1]*math.Pi/180 - yaw
		values = append(values, b(body.ObservedDeathAnimation), b(strings.Contains(path, "/monsters/")),
			b(strings.Contains(path, "players/")), b(body.Class == "misc_explobox"), b(path != ""),
			math.Sin(bodyYaw), math.Cos(bodyYaw), float64(body.Animation)/512)
		if len(values) != 20 {
			return nil, fmt.Errorf("invalid body feature width")
		}
		copy(result[at:at+20], values)
	}
	for _, value := range result {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf("nonfinite dynamic body feature")
		}
	}
	return result, nil
}
