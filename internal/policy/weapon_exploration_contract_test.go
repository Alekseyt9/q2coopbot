package policy

import (
	"math"
	"q2coopbot/internal/quake"
	"testing"
)

// Schema checks only; numerical validation and replay run on CUDA/native traces.
func TestWeaponExplorationContract(t *testing.T) {
	valid := func() PPOFile {
		return PPOFile{Features: EquippedWeaponFeatureVersion, WeaponHead: WeaponHeadVersion, AimModeHead: PrecisionHeadVersion, SpatialAim: &SpatialAimFile{}, WeaponExploration: &WeaponExplorationFile{Version: WeaponExplorationVersion, Temperature: 2}}
	}
	if err := validateWeaponExploration(PPOFile{}); err != nil {
		t.Fatal(err)
	}
	for _, temperature := range []float64{1, 2, 4} {
		f := valid()
		f.WeaponExploration.Temperature = temperature
		if err := validateWeaponExploration(f); err != nil {
			t.Fatal(err)
		}
	}
	for name, change := range map[string]func(*PPOFile){
		"version":   func(f *PPOFile) { f.WeaponExploration.Version = "unknown" },
		"feature":   func(f *PPOFile) { f.Features = WeaponFeatureVersion },
		"head":      func(f *PPOFile) { f.WeaponHead = "" },
		"precision": func(f *PPOFile) { f.AimModeHead = "" },
		"spatial":   func(f *PPOFile) { f.SpatialAim = nil },
		"nan":       func(f *PPOFile) { f.WeaponExploration.Temperature = math.NaN() },
		"inf":       func(f *PPOFile) { f.WeaponExploration.Temperature = math.Inf(1) },
		"small":     func(f *PPOFile) { f.WeaponExploration.Temperature = .9 },
		"large":     func(f *PPOFile) { f.WeaponExploration.Temperature = 4.1 },
	} {
		t.Run(name, func(t *testing.T) {
			f := valid()
			change(&f)
			if validateWeaponExploration(f) == nil {
				t.Fatal("invalid contract accepted")
			}
		})
	}
}

// Metadata and a scalar logit transform only; no neural policy is constructed.
func TestBlasterWeaponExplorationContract(t *testing.T) {
	for _, value := range []float64{1, 2, 4} {
		f := PPOFile{Features: ProjectileTypeNeutralFeatureVersion, WeaponHead: WeaponHeadVersion,
			AimModeHead: PrecisionHeadVersion, SpatialAim: &SpatialAimFile{},
			WeaponExploration: &WeaponExplorationFile{Version: BlasterWeaponExplorationVersion, Temperature: 2, BlasterTemperature: &value}}
		if err := validateWeaponExploration(f); err != nil {
			t.Fatal(err)
		}
	}
	for _, value := range []float64{0, .9, 4.1, math.NaN(), math.Inf(1)} {
		f := PPOFile{Features: EquippedWeaponFeatureVersion, WeaponHead: WeaponHeadVersion,
			AimModeHead: PrecisionHeadVersion, SpatialAim: &SpatialAimFile{},
			WeaponExploration: &WeaponExplorationFile{Version: BlasterWeaponExplorationVersion, Temperature: 2, BlasterTemperature: &value}}
		if validateWeaponExploration(f) == nil {
			t.Fatal("invalid blaster temperature admitted", value)
		}
	}
	f := PPOFile{Features: EquippedWeaponFeatureVersion, WeaponHead: WeaponHeadVersion,
		AimModeHead: PrecisionHeadVersion, SpatialAim: &SpatialAimFile{},
		WeaponExploration: &WeaponExplorationFile{Version: BlasterWeaponExplorationVersion, Temperature: 2}}
	if validateWeaponExploration(f) == nil {
		t.Fatal("v2 missing temperature admitted")
	}
	v := 1.0
	f.WeaponExploration = &WeaponExplorationFile{Version: WeaponExplorationVersion, Temperature: 2, BlasterTemperature: &v}
	if validateWeaponExploration(f) == nil {
		t.Fatal("legacy header admitted v2 field")
	}
}

func TestBlasterWeaponExplorationScalarGates(t *testing.T) {
	for _, scenario := range []string{"blaster", "unit", "stale", "missing", "no_ammo", "alone", "shotgun", "grenades", "unknown", "legacy_blaster"} {
		t.Run(scenario, func(t *testing.T) {
			value, age := 4.0, 0
			items := []quake.InventoryItem{{Name: "Blaster", Count: 1}, {Name: "Shotgun", Count: 1}, {Name: "Shells", Count: 20}}
			o := Observation{Weapon: "Blaster", Inventory: &items, InventoryAgeFrames: &age}
			p := PPO{file: PPOFile{WeaponExploration: &WeaponExplorationFile{
				Version: BlasterWeaponExplorationVersion, Temperature: 2, BlasterTemperature: &value}}}
			divisor := 4.0
			switch scenario {
			case "unit":
				value, divisor = 1, 1
			case "stale":
				age, divisor = 21, 1
			case "missing":
				o.Inventory, divisor = nil, 1
			case "no_ammo":
				items[2].Count, divisor = 0, 1
			case "alone":
				items, divisor = items[:1], 1
			case "shotgun":
				o.Weapon, divisor = "Shotgun", 1
			case "grenades":
				o.Weapon, divisor = "Grenades", 2
			case "unknown":
				o.Weapon, divisor = "unknown", 1
			case "legacy_blaster":
				p.file.WeaponExploration = &WeaponExplorationFile{Version: WeaponExplorationVersion, Temperature: 2}
				divisor = 1
			}
			raw := make([]float64, PrecisionOutputWidth)
			for i := range raw {
				raw[i] = float64(i) + .25
			}
			got, err := p.weaponExplorationRaw(o, raw)
			if err != nil {
				t.Fatal(err)
			}
			for i, original := range raw {
				want := float64(i) + .25
				if original != want {
					t.Fatal("input mutated")
				}
				if i >= 8 && i < 20 {
					want /= divisor
				}
				if got[i] != want {
					t.Fatalf("output %d changed unexpectedly: %v != %v", i, got[i], want)
				}
			}
		})
	}
}
