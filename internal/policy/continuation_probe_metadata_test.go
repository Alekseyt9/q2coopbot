package policy

import (
	"math"
	"math/rand"
	"testing"
)

// Pure metadata/scalar/PRNG checks. No neural network construction or forward.
func continuationMetadata() PPOFile {
	f := movementMetadata()
	f.SamplingSeed = 123
	f.DiagnosticContinuation = &DiagnosticContinuationFile{Version: DiagnosticContinuationVersion, Frame: 111, Seed: 456}
	return f
}

func TestDiagnosticContinuationMetadata(t *testing.T) {
	if err := validateDiagnosticContinuation(continuationMetadata()); err != nil {
		t.Fatal(err)
	}
	for _, scenario := range []string{"version", "frame", "large_frame", "seed", "large_seed", "phase", "deterministic", "features", "distribution", "head", "aim_probe", "execution"} {
		t.Run(scenario, func(t *testing.T) {
			f := continuationMetadata()
			switch scenario {
			case "version":
				f.DiagnosticContinuation.Version = "unknown"
			case "frame":
				f.DiagnosticContinuation.Frame = 1
			case "large_frame":
				f.DiagnosticContinuation.Frame = math.MaxInt32 + 1
			case "seed":
				f.DiagnosticContinuation.Seed = -1
			case "large_seed":
				f.DiagnosticContinuation.Seed = math.MaxInt32 + 1
			case "phase":
				f.DiagnosticContinuation.Frame++
			case "deterministic":
				f.Deterministic = true
			case "features":
				f.Features = FeatureVersion
			case "distribution":
				f.ActionDistribution = "unknown"
			case "head":
				f.AimCandidateHead = nil
			case "aim_probe":
				f.DiagnosticAimProbe = &DiagnosticAimProbeFile{}
			case "execution":
				f.ExecutionMode = "unknown"
			}
			if validateDiagnosticContinuation(f) == nil {
				t.Fatal("invalid continuation accepted")
			}
		})
	}
	f := continuationMetadata()
	f.DiagnosticMovementProbe = nil
	if validateDiagnosticContinuation(f) != nil {
		t.Fatal("fork-only reference rejected")
	}
	p := PPO{file: f}
	if !p.IsDiagnosticCombatProbe() || p.IsStochastic() {
		t.Fatal("fork admitted to ordinary PPO")
	}
}

func TestDiagnosticContinuationPRNG(t *testing.T) {
	p := PPO{file: continuationMetadata(), rng: rand.New(rand.NewSource(123))}
	original := rand.New(rand.NewSource(123))
	o := Observation{Identity: Identity{Life: 1, Frame: 110}}
	if err := p.beginContinuation(o); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 20; i++ {
		if p.rng.NormFloat64() != original.NormFloat64() {
			t.Fatal("prefix RNG changed")
		}
	}
	if p.continuationSamplingSeed(o) != 123 {
		t.Fatal("prefix hash seed changed")
	}
	o.Identity.Frame++
	if err := p.beginContinuation(o); err != nil {
		t.Fatal(err)
	}
	fork := rand.New(rand.NewSource(456))
	for i := 0; i < 20; i++ {
		if p.rng.NormFloat64() != fork.NormFloat64() {
			t.Fatal("wrong reset stream")
		}
	}
	if p.continuationSamplingSeed(o) != 456 {
		t.Fatal("fork hash seed absent")
	}
	o.Identity.Frame++
	if err := p.beginContinuation(o); err != nil {
		t.Fatal(err)
	}
	if p.rng.Float64() != fork.Float64() {
		t.Fatal("continuation reset twice")
	}
	o.Identity.Frame = 110
	if p.beginContinuation(o) == nil {
		t.Fatal("backward replay accepted")
	}
	p = PPO{file: continuationMetadata(), rng: rand.New(rand.NewSource(123))}
	o.Identity.Frame = 112
	if p.beginContinuation(o) == nil {
		t.Fatal("skipped fork accepted")
	}
}

func TestDiagnosticContinuationReceipt(t *testing.T) {
	p := PPO{file: continuationMetadata()}
	o := Observation{Identity: Identity{Life: 1, Frame: 110}}
	if p.validateContinuationSample(o, Sample{}) != nil {
		t.Fatal("prefix rejected")
	}
	o.Identity.Frame = 111
	r := &DiagnosticContinuationSample{Version: DiagnosticContinuationVersion, Frame: 111, Seed: 456, Reset: true, GaussianNoise: []float64{0, 1, -1, 2}, FireUniform: .5}
	if p.validateContinuationSample(o, Sample{ContinuationProbe: r}) != nil {
		t.Fatal("valid receipt rejected")
	}
	for _, scenario := range []string{"missing", "seed", "reset", "short", "noise", "fire", "life"} {
		t.Run(scenario, func(t *testing.T) {
			q := *r
			q.GaussianNoise = append([]float64(nil), r.GaussianNoise...)
			s := Sample{ContinuationProbe: &q}
			oo := o
			switch scenario {
			case "missing":
				s.ContinuationProbe = nil
			case "seed":
				q.Seed++
			case "reset":
				q.Reset = false
			case "short":
				q.GaussianNoise = q.GaussianNoise[:3]
			case "noise":
				q.GaussianNoise[0] = math.Inf(1)
			case "fire":
				q.FireUniform = 1
			case "life":
				oo.Identity.Life = 2
			}
			if p.validateContinuationSample(oo, s) == nil {
				t.Fatal("invalid receipt accepted")
			}
		})
	}
}
