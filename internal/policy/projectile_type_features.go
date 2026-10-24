package policy

import "sort"

// Experimental encoder only. PPO admission/CUDA migration must be added before
// these schemas can be used by a neural policy. Both preserve the exact v11
// body-ablated parent and add type bits aligned to its four projectile slots.
const ProjectileTypeFeatureVersion = "combat_features_v12_projectile_types"
const ProjectileTypeAblatedFeatureVersion = "combat_features_v12_projectile_types_ablated"
const ProjectileTypeNeutralFeatureVersion = "combat_features_v12_projectile_types_neutral"
const ProjectileTypeTailWidth = 4 * 5
const ProjectileTypeFeatureWidth = BodyFeatureWidth + ProjectileTypeTailWidth

func projectileTypeFeatures(o Observation) []float64 {
	result := make([]float64, ProjectileTypeTailWidth)
	if o.Projectiles == nil {
		return result
	}
	items := append([]Enemy(nil), (*o.Projectiles)...)
	// Same stable distance ordering as the existing environment features.
	sort.SliceStable(items, func(i, j int) bool { return items[i].Distance < items[j].Distance })
	for i, item := range items {
		if i == 4 {
			break
		}
		kind := 4 // Observed but unrecognized class remains explicit.
		switch item.Class {
		case "grenade":
			kind = 0
		case "hand_grenade":
			kind = 1
		case "rocket":
			kind = 2
		case "blaster_bolt":
			kind = 3
		}
		result[i*5+kind] = 1
	}
	return result
}
