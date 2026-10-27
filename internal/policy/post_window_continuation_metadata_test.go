package policy

import (
	"math"
	"math/rand"
	"testing"
)

func postWindowMetadata() PPOFile {
	return PPOFile{Features: MemoryRouteFeatureVersion, ActionDistribution: JointAimCandidateVersion,
		AimCandidateHead: &AimCandidateFile{}, SamplingSeed: 123,
		DiagnosticMovementProbe: &DiagnosticMovementProbeFile{Version: DiagnosticMovementWindowVersion, Frame: 374, Duration: 5, MeanShift: []float64{-1, 0}},
		DiagnosticContinuation:  &DiagnosticContinuationFile{Version: DiagnosticPostWindowContinuationVersion, Frame: 379, Seed: 98765}}
}

// Metadata, receipt and scalar PRNG only. No neural construction or inference.
func TestPostWindowContinuationMetadata(t *testing.T) {
	base := postWindowMetadata()
	if err := validateDiagnosticContinuation(base); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*PPOFile){
		func(f *PPOFile) { f.DiagnosticContinuation.Version = DiagnosticContinuationVersion },
		func(f *PPOFile) { f.DiagnosticContinuation.Frame = 375 },
		func(f *PPOFile) { f.DiagnosticContinuation.Frame = 380 },
		func(f *PPOFile) { f.Features = EquippedWeaponFeatureVersion },
		func(f *PPOFile) { f.DiagnosticMovementProbe = nil },
		func(f *PPOFile) { f.DiagnosticMovementProbe.Duration = 1 },
		func(f *PPOFile) {
			f.DiagnosticMovementProbe.Frame = math.MaxInt32 - 4
			f.DiagnosticContinuation.Frame = math.MaxInt32
		},
		func(f *PPOFile) { f.DiagnosticAnchorProbe = &DiagnosticAnchorProbeFile{} },
		func(f *PPOFile) { f.DiagnosticWeaponProbe = &DiagnosticWeaponProbeFile{} },
		func(f *PPOFile) { f.DiagnosticWideGate = &DiagnosticWideGateFile{} },
		func(f *PPOFile) { f.Deterministic = true },
	} {
		f := postWindowMetadata()
		change(&f)
		if validateDiagnosticContinuation(f) == nil {
			t.Fatal("invalid post-window fork admitted")
		}
	}
	p := PPO{file: base}
	if !p.IsDiagnosticCombatProbe() || p.IsStochastic() {
		t.Fatal("post-window fork eligible for ordinary PPO")
	}
}

func TestPostWindowContinuationPRNGAndReceipt(t *testing.T) {
	f := postWindowMetadata()
	p := PPO{file: f, rng: rand.New(rand.NewSource(123))}
	old := rand.New(rand.NewSource(123))
	for frame := 374; frame < 379; frame++ {
		o := Observation{Identity: Identity{Life: 1, Frame: frame}}
		if p.beginContinuation(o) != nil || p.rng.Float64() != old.Float64() || p.continuationSamplingSeed(o) != 123 {
			t.Fatal("movement-window RNG changed")
		}
	}
	o := Observation{Identity: Identity{Life: 1, Frame: 379}}
	if p.beginContinuation(o) != nil || p.rng.Float64() != rand.New(rand.NewSource(98765)).Float64() {
		t.Fatal("post-window stream reset differs")
	}
	r := &DiagnosticContinuationSample{Version: DiagnosticPostWindowContinuationVersion, Frame: 379, Seed: 98765,
		Reset: true, GaussianNoise: []float64{0, 0, 0, 0}, FireUniform: .5}
	if p.validateContinuationSample(o, Sample{ContinuationProbe: r}) != nil {
		t.Fatal("valid versioned receipt rejected")
	}
	r.Version = DiagnosticContinuationVersion
	if p.validateContinuationSample(o, Sample{ContinuationProbe: r}) == nil {
		t.Fatal("v1 receipt admitted to v2")
	}
	if p.validateMovementSample(379, Sample{}) != nil {
		t.Fatal("completed movement window still requires movement receipt")
	}
}
