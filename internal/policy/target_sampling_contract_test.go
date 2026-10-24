package policy

import (
	"math"
	"testing"
)

func TestTargetSamplingContract(t *testing.T) {
	f := PPOFile{Features: EquippedWeaponFeatureVersion, TargetHead: TargetHeadVersion,
		ActionDistribution: JointAimCandidateVersion, AimCandidateHead: &AimCandidateFile{}}
	if err := validateTargetSampling(f); err != nil {
		t.Fatal(err)
	}
	for _, value := range []float64{.25, .5, 1, 2} {
		f.TargetSampling = &TargetSamplingFile{Version: TargetSamplingVersion, Temperature: value}
		if err := validateTargetSampling(f); err != nil {
			t.Fatal(err)
		}
	}
	for _, value := range []float64{0, -.5, .249, 2.001, math.NaN(), math.Inf(1)} {
		f.TargetSampling.Temperature = value
		if err := validateTargetSampling(f); err == nil {
			t.Fatalf("invalid temperature accepted: %v", value)
		}
	}
	f.TargetSampling.Temperature = .5
	f.TargetSampling.Version = "unknown"
	if err := validateTargetSampling(f); err == nil {
		t.Fatal("unknown version accepted")
	}
	f.TargetSampling.Version = TargetSamplingVersion
	f.ActionDistribution = ""
	if err := validateTargetSampling(f); err == nil {
		t.Fatal("unsupported scorer accepted")
	}
}
