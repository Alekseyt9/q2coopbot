package policy

import (
	"math"
	"testing"

	"q2coopbot/internal/quake"
)

func weaponProbeMetadata() PPOFile {
	return PPOFile{Features: ProjectileTypeNeutralFeatureVersion, WeaponHead: WeaponHeadVersion,
		TargetHead: TargetHeadVersion, ActionDistribution: JointAimCandidateVersion,
		AimCandidateHead: &AimCandidateFile{}, ConditionalFire: &SpatialAimFile{},
		DiagnosticWeaponProbe: &DiagnosticWeaponProbeFile{Version: DiagnosticWeaponProbeVersion,
			Frame: 98, Mode: "identity", ExpectedEquippedWeapon: 1, ForcedWeapon: 2}}
}

// Metadata/scalar checks only: no neural network is constructed or evaluated.
func TestDiagnosticWeaponProbeMetadata(t *testing.T) {
	for _, mode := range []string{"identity", "force"} {
		f := weaponProbeMetadata()
		f.DiagnosticWeaponProbe.Mode = mode
		if err := validateDiagnosticWeaponProbe(f); err != nil {
			t.Fatal(err)
		}
	}
	for _, change := range []func(*PPOFile){
		func(f *PPOFile) { f.DiagnosticWeaponProbe.Version = "unknown" },
		func(f *PPOFile) { f.DiagnosticWeaponProbe.Frame = 0 },
		func(f *PPOFile) { f.DiagnosticWeaponProbe.Frame = math.MaxInt32 + 1 },
		func(f *PPOFile) { f.DiagnosticWeaponProbe.Mode = "keep" },
		func(f *PPOFile) { f.DiagnosticWeaponProbe.ForcedWeapon = 0 },
		func(f *PPOFile) { f.DiagnosticWeaponProbe.ExpectedEquippedWeapon = 2 },
		func(f *PPOFile) { f.DiagnosticMovementProbe = &DiagnosticMovementProbeFile{} },
		func(f *PPOFile) { f.DiagnosticContinuation = &DiagnosticContinuationFile{} },
		func(f *PPOFile) { f.DiagnosticAimProbe = &DiagnosticAimProbeFile{} },
		func(f *PPOFile) { f.Deterministic = true },
		func(f *PPOFile) { f.WeaponHead = "" },
		func(f *PPOFile) { f.TargetHead = "" },
		func(f *PPOFile) { f.ConditionalFire = nil },
		func(f *PPOFile) { f.Features = ThreatFeatureVersion },
		func(f *PPOFile) { f.ExecutionMode = "greedy" },
	} {
		f := weaponProbeMetadata()
		change(&f)
		if validateDiagnosticWeaponProbe(f) == nil {
			t.Fatal("invalid metadata admitted")
		}
	}
	f := weaponProbeMetadata()
	p := &PPO{file: f}
	if !p.IsDiagnosticCombatProbe() {
		t.Fatal("probe admitted as ordinary combat")
	}
}

func weaponProbeObservation() Observation {
	age := 0
	items := []quake.InventoryItem{{Name: "Shotgun", Count: 1}, {Name: "Shells", Count: 5}}
	return Observation{Identity: Identity{Life: 1, Frame: 98}, Weapon: "Blaster",
		InventoryAgeFrames: &age, Inventory: &items, Enemies: []Enemy{{ID: 1, Distance: 100}}}
}

func TestDiagnosticWeaponProbeScalarGuards(t *testing.T) {
	p := &PPO{file: weaponProbeMetadata()}
	o := weaponProbeObservation()
	source := make([]float64, 12)
	source[0], source[2] = .25, .75
	got, err := p.weaponProbeProbabilities(o, 1, source)
	if err != nil || &got[0] != &source[0] {
		t.Fatal("identity modified source CDF", err)
	}
	p.file.DiagnosticWeaponProbe.Mode = "force"
	got, err = p.weaponProbeProbabilities(o, 1, source)
	if err != nil || got[2] != 1 || got[0] != 0 || source[0] != .25 {
		t.Fatal("force modified source", err)
	}
	for _, change := range []func(*Observation){
		func(o *Observation) { o.Identity.Life = 2 },
		func(o *Observation) { o.Weapon = "Shotgun" },
		func(o *Observation) { o.InventoryAgeFrames = nil },
		func(o *Observation) { age := 21; o.InventoryAgeFrames = &age },
		func(o *Observation) { o.Inventory = nil },
		func(o *Observation) { o.Enemies = nil },
	} {
		x := weaponProbeObservation()
		change(&x)
		if _, err := p.weaponProbeProbabilities(x, 1, source); err == nil {
			t.Fatal("bad root admitted")
		}
	}
	for _, frame := range []int{97, 99} {
		o.Identity.Frame = frame
		got, err = p.weaponProbeProbabilities(o, 0, source)
		if err != nil || &got[0] != &source[0] {
			t.Fatal("intervened outside declared frame")
		}
	}
}

func TestDiagnosticWeaponProbeScalarReceipt(t *testing.T) {
	p := &PPO{file: weaponProbeMetadata()}
	o := weaponProbeObservation()
	for _, mode := range []string{"identity", "force"} {
		p.file.DiagnosticWeaponProbe.Mode = mode
		for _, uniform := range []float64{0, .4, .999999} {
			logs := make([]float64, 12)
			for i := range logs {
				logs[i] = math.Inf(-1)
			}
			logs[0], logs[2] = math.Log(.25), math.Log(.75)
			source := make([]float64, 12)
			for i, v := range logs {
				source[i] = math.Exp(v)
			}
			applied, err := p.weaponProbeProbabilities(o, 1, source)
			if err != nil {
				t.Fatal(err)
			}
			chosen := 2
			if mode == "identity" && uniform <= .25 {
				chosen = 0
			}
			s := Sample{Target: 1, Weapon: chosen, WeaponProbe: &DiagnosticWeaponSample{
				Version: DiagnosticWeaponProbeVersion, Frame: 98, Mode: mode,
				SourceProbabilities: source, AppliedProbabilities: applied, WeaponUniform: uniform,
				FireUniform: .5, GaussianNoise: []float64{0, 0, 0, 0}}}
			got, err := p.reviewWeaponProbe(o, s, logs)
			if err != nil {
				t.Fatal(err)
			}
			if mode == "force" && got[2] != 0 {
				t.Fatal("forced category log density differs")
			}
			s.WeaponProbe.WeaponUniform = math.NaN()
			if _, err := p.reviewWeaponProbe(o, s, logs); err == nil {
				t.Fatal("nonfinite receipt admitted")
			}
		}
	}
}
