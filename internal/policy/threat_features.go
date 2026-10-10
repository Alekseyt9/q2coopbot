package policy

import (
	"fmt"
	"math"
)

const ThreatFeatureVersion = "combat_features_v9"
const ThreatFeatureWidth = NavigationFeatureWidth + 8*30

// Eight recency/ID sorted slots: present, age, 21 types, last observed XYZ,
// velocity present, last observed velocity XYZ. Visible-target masks in the
// unchanged v8 prefix still exclude occluded remembered monsters.
func threatFeatures(o Observation) ([]float64, error) {
	if len(monsterTypes) != 21 || len(o.RememberedThreats) > 8 {
		return nil, fmt.Errorf("threat memory dimensions")
	}
	v := make([]float64, 240)
	yaw := degrees(o.ViewAngles[1]) * math.Pi / 180
	seen := map[int]bool{}
	for slot, e := range o.RememberedThreats {
		if e.ID <= 0 || seen[e.ID] || e.AgeFrames < 0 || e.AgeFrames > ThreatMemoryFrames {
			return nil, fmt.Errorf("invalid remembered threat identity/age")
		}
		seen[e.ID] = true
		at := slot * 30
		v[at], v[at+1] = 1, float64(e.AgeFrames)/ThreatMemoryFrames
		kind := MonsterType(e.Class, e.ModelPath)
		for i, t := range monsterTypes {
			if t == kind {
				v[at+2+i] = 1
			}
		}
		for i, vec := range []*[3]float64{(*[3]float64)(&e.Relative), (*[3]float64)(e.Velocity)} {
			if vec == nil {
				continue
			}
			for _, x := range vec {
				if math.IsNaN(x) || math.IsInf(x, 0) {
					return nil, fmt.Errorf("nonfinite remembered threat")
				}
			}
			start, scale := at+23, 512.
			if i == 1 {
				v[at+26] = 1
				start, scale = at+27, 400
			}
			local := [3]float64{vec[0]*math.Cos(yaw) + vec[1]*math.Sin(yaw), -vec[0]*math.Sin(yaw) + vec[1]*math.Cos(yaw), vec[2]}
			for axis, x := range local {
				v[start+axis] = math.Max(-4, math.Min(4, x/scale))
			}
		}
	}
	return v, nil
}
