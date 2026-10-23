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
