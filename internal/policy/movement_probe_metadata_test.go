package policy

import (
	"math"
	"testing"
)

func movementMetadata() PPOFile {
	return PPOFile{Features: EquippedWeaponFeatureVersion, ActionDistribution: JointAimCandidateVersion,
		AimCandidateHead: &AimCandidateFile{}, DiagnosticMovementProbe: &DiagnosticMovementProbeFile{
			Version: DiagnosticMovementProbeVersion, Frame: 110, MeanShift: []float64{0, 1}}}
}

// The intervention changes output movement means, not the appended input schema.
// These are admission checks only; no network is constructed or evaluated.
func TestDiagnosticMovementEquippedPrefixSchemas(t *testing.T) {
	for _, version := range []string{EquippedWeaponFeatureVersion, BodyFeatureVersion, BodyAblatedFeatureVersion,
		ProjectileTypeFeatureVersion, ProjectileTypeNeutralFeatureVersion, ProjectileTypeAblatedFeatureVersion} {
		f := movementMetadata()
		f.Features = version
		if err := validateDiagnosticMovementProbe(f); err != nil {
			t.Fatalf("%s one frame: %v", version, err)
		}
		f.DiagnosticMovementProbe.Version = DiagnosticMovementWindowVersion
		f.DiagnosticMovementProbe.Duration = 3
		if err := validateDiagnosticMovementProbe(f); err != nil {
			t.Fatalf("%s window: %v", version, err)
		}
	}
	for _, version := range []string{FeatureVersion, ThreatFeatureVersion, "combat_features_v999"} {
		f := movementMetadata()
		f.Features = version
		if validateDiagnosticMovementProbe(f) == nil {
			t.Fatalf("unsupported feature schema admitted: %s", version)
		}
	}
}

// Metadata and scalar means only; no network is constructed or evaluated.
func TestDiagnosticMovementProbeMetadata(t *testing.T) {
	f := movementMetadata()
	if err := validateDiagnosticMovementProbe(f); err != nil {
		t.Fatal(err)
	}
	for _, scenario := range []string{"version", "frame", "large_frame", "short", "wide", "nan", "inf", "range", "deterministic", "features", "distribution", "head", "combined", "execution"} {
		t.Run(scenario, func(t *testing.T) {
			f := movementMetadata()
			switch scenario {
			case "version":
				f.DiagnosticMovementProbe.Version = "unknown"
			case "frame":
				f.DiagnosticMovementProbe.Frame = 0
			case "large_frame":
				f.DiagnosticMovementProbe.Frame = math.MaxInt32 + 1
			case "short":
				f.DiagnosticMovementProbe.MeanShift = []float64{1}
			case "wide":
				f.DiagnosticMovementProbe.MeanShift = []float64{0, 1, 0}
			case "nan":
				f.DiagnosticMovementProbe.MeanShift[0] = math.NaN()
			case "inf":
				f.DiagnosticMovementProbe.MeanShift[0] = math.Inf(1)
			case "range":
				f.DiagnosticMovementProbe.MeanShift[0] = 2.01
			case "deterministic":
				f.Deterministic = true
			case "features":
				f.Features = FeatureVersion
			case "distribution":
				f.ActionDistribution = "unknown"
			case "head":
				f.AimCandidateHead = nil
			case "combined":
				f.DiagnosticAimProbe = &DiagnosticAimProbeFile{}
			case "execution":
				f.ExecutionMode = "unknown"
			}
			if validateDiagnosticMovementProbe(f) == nil {
				t.Fatal("invalid probe accepted")
			}
		})
	}
}

func TestDiagnosticMovementMeansScalar(t *testing.T) {
	f := movementMetadata()
	p := PPO{file: f}
	means := [4]float64{math.Copysign(0, -1), .3, -.2, .1}
	for _, frame := range []int{109, 111} {
		got := p.movementProbeMeans(frame, means)
		for i := range means {
			if math.Float64bits(got[i]) != math.Float64bits(means[i]) {
				t.Fatal("outside-frame change")
			}
		}
	}
	f.DiagnosticMovementProbe.MeanShift = []float64{-1, 1}
	got := p.movementProbeMeans(110, means)
	if got != [4]float64{-1, 1.3, -.2, .1} {
		t.Fatal("wrong components", got)
	}
	f.DiagnosticMovementProbe.MeanShift = []float64{0, 0}
	got = p.movementProbeMeans(110, means)
	for i := range means {
		if math.Float64bits(got[i]) != math.Float64bits(means[i]) {
			t.Fatal("zero-control change")
		}
	}
	if !p.IsDiagnosticCombatProbe() || p.IsStochastic() {
		t.Fatal("diagnostic probe admitted as ordinary stochastic PPO")
	}
}

func TestDiagnosticMovementDrawReceipt(t *testing.T) {
	p := PPO{file: movementMetadata()}
	if p.validateMovementSample(110, Sample{}) == nil {
		t.Fatal("missing draw evidence accepted")
	}
	s := Sample{MovementProbe: &DiagnosticMovementSample{Version: DiagnosticMovementProbeVersion, Frame: 110,
		MeanShift: []float64{0, 1}, GaussianNoise: []float64{.1, -.1, .001, 0}, FireUniform: .5}}
	if err := p.validateMovementSample(110, s); err != nil {
		t.Fatal(err)
	}
	if p.validateMovementSample(109, s) == nil {
		t.Fatal("receipt outside intervention frame")
	}
	s.MovementProbe.GaussianNoise[2] = math.NaN()
	if p.validateMovementSample(110, s) == nil {
		t.Fatal("invalid draw accepted")
	}
}

// Exercise window boundaries and receipts without constructing a neural model.
func TestDiagnosticMovementWindowMetadataAndScalars(t *testing.T) {
	f := movementMetadata()
	f.DiagnosticMovementProbe.Version = DiagnosticMovementWindowVersion
	f.DiagnosticMovementProbe.Duration = 5
	if err := validateDiagnosticMovementProbe(f); err != nil {
		t.Fatal(err)
	}
	p := PPO{file: f}
	means := [4]float64{.2, .3, -.2, .1}
	for frame := 109; frame <= 115; frame++ {
		want := means
		inside := frame >= 110 && frame < 115
		if inside {
			want[1] += 1
		}
		if got := p.movementProbeMeans(frame, means); got != want {
			t.Fatalf("frame %d: %v != %v", frame, got, want)
		}
		s := Sample{}
		if inside {
			s.MovementProbe = &DiagnosticMovementSample{Version: DiagnosticMovementWindowVersion, Frame: frame,
				MeanShift: []float64{0, 1}, GaussianNoise: []float64{0, 0, 0, 0}, FireUniform: .5}
		}
		if err := p.validateMovementSample(frame, s); err != nil {
			t.Fatal(frame, err)
		}
		if inside {
			s.MovementProbe.Frame = 109
			if p.validateMovementSample(frame, s) == nil {
				t.Fatal("wrong receipt frame accepted")
			}
		}
	}
	for _, duration := range []int{-1, 0, 1, 11} {
		f.DiagnosticMovementProbe.Duration = duration
		if validateDiagnosticMovementProbe(f) == nil {
			t.Fatal("invalid duration accepted", duration)
		}
	}
	f.DiagnosticMovementProbe.Duration = 5
	f.DiagnosticMovementProbe.Frame = math.MaxInt32 - 3
	if validateDiagnosticMovementProbe(f) == nil {
		t.Fatal("overflowing window accepted")
	}
	f = movementMetadata()
	f.DiagnosticMovementProbe.Duration = 5
	if validateDiagnosticMovementProbe(f) == nil {
		t.Fatal("legacy probe accepted window metadata")
	}
}
