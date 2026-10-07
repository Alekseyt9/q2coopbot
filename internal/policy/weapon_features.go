package policy

import (
	"fmt"
	"math"
)

// WeaponFeatureVersion preserves all 814 v5 features, then appends 31
// inventory-only values: known/fresh, 11 owned flags, 6 ammo quantities,
// and 12 action availability flags. Unknown inventory never invents stock.
const WeaponFeatureVersion = "combat_features_v6"
const WeaponFeatureOffset = 814
const WeaponMaskOffset = 833
const WeaponFeatureWidth = 845

// Index zero keeps the equipped weapon; ordering is a serialized contract.
var weaponNames = [...]string{"", "Blaster", "Shotgun", "Super Shotgun", "Machinegun", "Chaingun", "Grenade Launcher", "Rocket Launcher", "HyperBlaster", "Railgun", "BFG10K", "Grenades"}
var ammoNames = [...]string{"Bullets", "Shells", "Cells", "Grenades", "Rockets", "Slugs"}
var ammoScale = [...]float64{200, 100, 200, 50, 50, 50}
var weaponAmmo = [...]string{"", "", "Shells", "Shells", "Bullets", "Bullets", "Grenades", "Rockets", "Cells", "Slugs", "Cells", "Grenades"}
var weaponMinimum = [...]int{0, 0, 1, 2, 1, 1, 1, 1, 1, 1, 50, 1}

func WeaponChoices() []string { return append([]string(nil), weaponNames[:]...) }

// WeaponAvailability uses svc_inventory and CS_ITEMS only. Keep is always
// permitted, including during a stale/unknown inventory interval. Masks are
// conservative stock ammo minima, not a promise of cooldown/fire success.
func WeaponAvailability(o Observation) ([]bool, error) {
	_, mask, err := weaponFeatures(o)
	return mask, err
}

func weaponFeatures(o Observation) ([]float64, []bool, error) {
	v := make([]float64, WeaponFeatureWidth-WeaponFeatureOffset)
	mask := make([]bool, len(weaponNames))
	mask[0] = true
	v[19] = 1
	if o.Inventory == nil {
		return v, mask, nil
	}
	v[0] = 1
	counts := map[string]int{}
	relevant := map[string]bool{}
	for _, name := range weaponNames[1:] {
		relevant[name] = true
	}
	for _, name := range ammoNames {
		relevant[name] = true
	}
	for _, item := range *o.Inventory {
		if !relevant[item.Name] {
			continue
		}
		if item.Count < 0 {
			return nil, nil, fmt.Errorf("negative observed inventory count")
		}
		if _, exists := counts[item.Name]; exists {
			return nil, nil, fmt.Errorf("duplicate observed inventory item")
		}
		counts[item.Name] = item.Count
	}
	for i, name := range weaponNames[1:] {
		if counts[name] > 0 {
			v[2+i] = 1
		}
	}
	for i, name := range ammoNames {
		v[13+i] = math.Min(4, float64(counts[name])/ammoScale[i])
	}
	if o.InventoryAgeFrames == nil || *o.InventoryAgeFrames < 0 || *o.InventoryAgeFrames > 20 {
		return v, mask, nil
	}
	v[1] = 1
	for i, name := range weaponNames[1:] {
		j := i + 1
		mask[j] = counts[name] > 0 && (weaponAmmo[j] == "" || counts[weaponAmmo[j]] >= weaponMinimum[j])
		if mask[j] {
			v[19+j] = 1
		}
	}
	return v, mask, nil
}
