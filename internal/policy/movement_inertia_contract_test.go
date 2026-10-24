package policy

import (
	"math"
	"testing"
)

func TestMovementInertiaContract(t *testing.T) {
	base := PPOFile{Features: EquippedWeaponFeatureVersion, SpatialAim: &SpatialAimFile{}}
	if err := validateMovementInertia(base); err != nil {
		t.Fatal(err)
	}
	for _, rho := range []float64{0, .65, .95} {
		f := base
		f.MovementInertia = &MovementInertiaFile{Version: MovementInertiaVersion, Rho: rho}
		if err := validateMovementInertia(f); err != nil {
			t.Fatal(err)
		}
	}
	for _, rho := range []float64{-.01, .96, math.NaN(), math.Inf(1)} {
		f := base
		f.MovementInertia = &MovementInertiaFile{Version: MovementInertiaVersion, Rho: rho}
		if err := validateMovementInertia(f); err == nil {
			t.Fatal("invalid rho accepted")
		}
	}
	f := base
	f.MovementInertia = &MovementInertiaFile{Version: "unknown", Rho: .65}
	if err := validateMovementInertia(f); err == nil {
		t.Fatal("unknown version accepted")
	}
	f.MovementInertia.Version = MovementInertiaVersion
	f.Features = FeatureVersion
	if err := validateMovementInertia(f); err == nil {
		t.Fatal("unsupported input contract accepted")
	}
}
