package policy

import (
	"math"
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
