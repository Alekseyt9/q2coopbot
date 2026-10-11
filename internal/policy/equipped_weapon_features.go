package policy

import (
	"fmt"
	"math"
	"strings"
)

const EquippedWeaponFeatureVersion = "combat_features_v10"
const EquippedWeaponFeatureWidth = ThreatFeatureWidth + 26

var equippedWeaponModels = [...]string{"v_blast", "v_shotg", "v_shotg2", "v_machn", "v_chain", "v_launch", "v_rocket", "v_hyperb", "v_rail", "v_bfg", "v_handgr"}

func equippedWeaponIndex(name string) int {
	name = strings.ToLower(strings.ReplaceAll(strings.TrimSpace(name), "\\", "/"))
	for index, label := range weaponNames[1:] {
		if name == strings.ToLower(label) || name == "models/weapons/"+equippedWeaponModels[index]+"/tris.md2" {
			return index + 1
		}
	}
	return 0
}

// Current weapon and previously queued local use request are distinct facts.
// Neither inventory nor an imagined server pending state fills absent fields.
func equippedWeaponFeatures(o Observation) ([]float64, error) {
	v := make([]float64, 26)
	v[equippedWeaponIndex(o.Weapon)] = 1
	v[12] = 1 // no observed own request
	if o.PreviousWeaponRequest == nil && o.PreviousWeaponRequestAgeFrames == nil {
		return v, nil
	}
	if o.PreviousWeaponRequest == nil || o.PreviousWeaponRequestAgeFrames == nil || *o.PreviousWeaponRequestAgeFrames < 0 {
		return nil, fmt.Errorf("incomplete or invalid own weapon request")
	}
	index := equippedWeaponIndex(*o.PreviousWeaponRequest)
	if index == 0 {
		return nil, fmt.Errorf("unknown own weapon request")
	}
	v[12], v[12+index], v[24] = 0, 1, 1
	v[25] = math.Min(4, float64(*o.PreviousWeaponRequestAgeFrames)/30)
	return v, nil
}
