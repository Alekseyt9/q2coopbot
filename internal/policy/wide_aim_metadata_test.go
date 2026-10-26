package policy

import (
	"math"
	"testing"

	"q2coopbot/internal/quake"
)

func TestWideAimMetadataSourceAndGeometry(t *testing.T) {
	visible, solid := true, uint16(2|(3<<5)|(8<<10))
	kick := quake.Vec3{2, 3, 0}
	o := Observation{Identity: Identity{Life: 1, Frame: 123}, Health: 100, Weapon: "Machinegun", KickAngles: &kick,
		Enemies: []Enemy{{ID: 17, Distance: 100, Relative: quake.Vec3{100, 0, 0}, ClearShot: &visible, Solid: &solid}}}
	p := &PPO{file: PPOFile{SamplingSeed: 7, WideAim: &WideAimFile{Enabled: true}}}
	s := Sample{Target: 1, Latent: [4]float64{.3, -.4, .1, -.2}, Attack: true, Vertical: 1}
	a, active, err := observedWideAim(o, s)
	if err != nil || !active || math.Abs(a[0]+3) > 1e-12 || math.Abs(a[1]-(-math.Atan2(8, 100)*180/math.Pi-2)) > 1e-12 {
		t.Fatal(a, active, err)
	}
	for _, logit := range []float64{-20, 20} {
		r := &WideAimSample{Version: WideAimVersion, SourceLatent: s.Latent, Anchor: a, Uniform: WideAimUniform(7, 123, 1), Logit: logit}
		if r.Uniform < math.Exp(-softplus(-logit)) {
			r.Choice = 1
		}
		r.LogProbability = wideLogProbability(logit, r.Choice)
		v := s
		v.WideAim = r
		v.Latent, err = applyWideChoice(s.Latent, a, r.Choice)
		if err != nil {
			t.Fatal(err)
		}
		got, err := p.legacyWideSample(o, v)
		if err != nil || got != s || v.Latent[0] != s.Latent[0] || v.Latent[1] != s.Latent[1] {
			t.Fatal("source event not recovered", err)
		}
		if r.Choice == 0 && v.Latent != s.Latent {
			t.Fatal("keep path not exact")
		}
		for _, mutate := range []func(*Sample){
			func(s *Sample) { s.WideAim.Uniform += .001 },
			func(s *Sample) { s.WideAim.Choice = 2 },
			func(s *Sample) { s.WideAim.Anchor[0] += .001 },
			func(s *Sample) { s.WideAim.LogProbability += .001 },
			func(s *Sample) { s.WideAim.Logit = math.NaN() },
			func(s *Sample) { s.Latent[0] += .001 },
		} {
			bad := v
			copyReceipt := *r
			bad.WideAim = &copyReceipt
			mutate(&bad)
			if _, err := p.legacyWideSample(o, bad); err == nil {
				t.Fatal("tampered event accepted")
			}
		}
		bad := v
		bad.WideAim = nil
		if _, err := p.legacyWideSample(o, bad); err == nil {
			t.Fatal("missing wide event accepted")
		}
		oo := o
		oo.Identity.Frame++
		if _, err := p.legacyWideSample(oo, v); err == nil {
			t.Fatal("stale uniform accepted")
		}
		oo = o
		oo.Weapon = "Blaster"
		if _, err := p.legacyWideSample(oo, v); err == nil {
			t.Fatal("event outside MG accepted")
		}
	}
	p.file.WideAim.Enabled = false
	if got, err := p.legacyWideSample(o, s); err != nil || got != s {
		t.Fatal("disabled path changed", err)
	}
	p.file.WideAim = nil
	if got, err := p.legacyWideSample(o, s); err != nil || got != s {
		t.Fatal("legacy path changed", err)
	}
}

func TestWideAimHeaderAndRNGMetadata(t *testing.T) {
	f := PPOFile{CombatExecution: StockWeaponsFireCombatExecution, Features: EquippedWeaponFeatureVersion, TargetHead: TargetHeadVersion,
		AimModeHead: PrecisionHeadVersion, ActionDistribution: WideAimDistribution, AimCandidateHead: &AimCandidateFile{}, ConditionalFire: &SpatialAimFile{},
		WideAim: &WideAimFile{Version: WideAimVersion, Sampler: WideAimSampler, Enabled: true}}
	w := WideAimInputWidth
	for _, n := range []int{32, 16, 1} {
		l := DenseLayer{Weight: make([][]float64, n), Bias: make([]float64, n)}
		for i := range l.Weight {
			l.Weight[i] = make([]float64, w)
		}
		f.WideAim.Layers = append(f.WideAim.Layers, l)
		w = n
	}
	if err := validateWideAim(f); err != nil {
		t.Fatal(err)
	}
	p := &PPO{file: f}
	if !p.IsStochastic() || p.IsDiagnosticCombatProbe() {
		t.Fatal("learned event incorrectly diagnostic")
	}
	for _, change := range []func(*PPOFile){
		func(f *PPOFile) { f.ActionDistribution = JointAimCandidateVersion }, func(f *PPOFile) { f.WideAim = nil },
		func(f *PPOFile) { f.Deterministic = true }, func(f *PPOFile) { f.ConditionalFire = nil },
		func(f *PPOFile) { f.DiagnosticAnchorProbe = &DiagnosticAnchorProbeFile{} },
	} {
		v := f
		change(&v)
		if validateWideAim(v) == nil {
			t.Fatal("bad header accepted")
		}
	}
	for _, frame := range []int{98, 123, 324} {
		u := WideAimUniform(7, frame, 1)
		if u < 0 || u >= 1 || u != WideAimUniform(7, frame, 1) || u == WideAimUniform(7, frame, 2) {
			t.Fatal("bad deterministic uniform")
		}
	}
}
