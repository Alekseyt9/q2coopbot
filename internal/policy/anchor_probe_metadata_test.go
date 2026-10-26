package policy

import (
	"math"
	"reflect"
	"testing"

	"q2coopbot/internal/quake"
)

// No neural forward pass: these tests cover only metadata and geometry.
func TestAnchorProbeGeometryAndReceipt(t *testing.T) {
	visible, solid := true, uint16(2|(3<<5)|(8<<10))
	kick := quake.Vec3{2, 3, 0}
	o := Observation{Identity: Identity{Life: 1, Frame: 123}, Health: 100, Weapon: "Machinegun", KickAngles: &kick,
		Enemies: []Enemy{{ID: 17, Distance: 100, Relative: quake.Vec3{100, 0, 0}, ClearShot: &visible, Solid: &solid}}}
	s := Sample{Target: 1, Latent: [4]float64{.3, -.4, .1, -.2}, Attack: true, Vertical: 1}
	p := &PPO{file: PPOFile{DiagnosticAnchorProbe: &DiagnosticAnchorProbeFile{Version: DiagnosticAnchorProbeVersion, Frame: 123, Mode: "observed_mg"}}}
	original := s
	r, err := p.anchorReceipt(o, s)
	if err != nil {
		t.Fatal(err)
	}
	if s != original {
		t.Fatal("source sample mutated")
	}
	// bbox AimPoint=22, standing centered muzzle=14, yaw=-kick yaw.
	want := [2]float64{-3, -math.Atan2(8, 100)*180/math.Pi - 2}
	for i := range want {
		if math.Abs(r.Returned[i]-want[i]) > 1e-12 {
			t.Fatal(r.Returned, want)
		}
	}
	s.AnchorProbe = r
	a := Action{TargetEntity: 17, Forward: .5, Side: -.7, Attack: true, Vertical: "jump", Weapon: "keep", YawDelta: r.Source[0], PitchDelta: r.Source[1]}
	b, err := p.applyAnchorReceipt(o, s, a)
	if err != nil {
		t.Fatal(err)
	}
	a.YawDelta, a.PitchDelta = want[0], want[1]
	if !reflect.DeepEqual(a, b) {
		t.Fatal("non-angular action changed", a, b)
	}
	r.Returned[0] += .001
	if _, err := p.applyAnchorReceipt(o, s, b); err == nil {
		t.Fatal("tampered receipt accepted")
	}
	s.AnchorProbe = nil
	if _, err := p.applyAnchorReceipt(o, s, a); err == nil {
		t.Fatal("missing receipt accepted")
	}
	o.Identity.Frame++
	if got, err := p.applyAnchorReceipt(o, s, a); err != nil || got != a {
		t.Fatal("changed outside branch", err)
	}
	s.AnchorProbe = r
	if _, err := p.applyAnchorReceipt(o, s, a); err == nil {
		t.Fatal("receipt outside branch accepted")
	}
	o.Identity.Frame--
	p.file.DiagnosticAnchorProbe.Mode = "identity"
	s.AnchorProbe, err = p.anchorReceipt(o, s)
	if err != nil {
		t.Fatal(err)
	}
	a.YawDelta, a.PitchDelta = s.AnchorProbe.Source[0], s.AnchorProbe.Source[1]
	if got, err := p.applyAnchorReceipt(o, s, a); err != nil || got != a {
		t.Fatal("identity not exact", err)
	}
	o.Ducked = true
	p.file.DiagnosticAnchorProbe.Mode = "observed_mg"
	r, err = p.anchorReceipt(o, s)
	if err != nil || math.Abs(r.Returned[1]-(-math.Atan2(32, 100)*180/math.Pi-2)) > 1e-12 {
		t.Fatal("ducked muzzle differs", r, err)
	}
	for _, change := range []func(*Observation, *Sample){
		func(o *Observation, s *Sample) { o.KickAngles = nil },
		func(o *Observation, s *Sample) { o.Health = 0 },
		func(o *Observation, s *Sample) { o.Identity.Life = 2 },
		func(o *Observation, s *Sample) { o.Weapon = "Blaster" },
		func(o *Observation, s *Sample) { s.Target = 0 },
		func(o *Observation, s *Sample) { s.AimMode = 1 },
	} {
		oo, ss := o, s
		change(&oo, &ss)
		if _, err := p.anchorReceipt(oo, ss); err == nil {
			t.Fatal("unsupported root accepted")
		}
	}
}

func TestAnchorProbeHeaderAndTrainingExclusion(t *testing.T) {
	f := PPOFile{Features: EquippedWeaponFeatureVersion, TargetHead: TargetHeadVersion, WeaponHead: WeaponHeadVersion,
		AimModeHead: PrecisionHeadVersion, ActionDistribution: JointAimCandidateVersion, AimCandidateHead: &AimCandidateFile{},
		DiagnosticAnchorProbe: &DiagnosticAnchorProbeFile{Version: DiagnosticAnchorProbeVersion, Frame: 123, Mode: "identity"}}
	if err := validateDiagnosticAnchorProbe(f); err != nil {
		t.Fatal(err)
	}
	p := &PPO{file: f}
	if !p.IsDiagnosticCombatProbe() || p.IsStochastic() {
		t.Fatal("diagnostic admitted to ordinary PPO")
	}
	for _, change := range []func(*PPOFile){
		func(f *PPOFile) { f.Deterministic = true },
		func(f *PPOFile) { f.AimCandidateHead = nil },
		func(f *PPOFile) { f.DiagnosticWeaponProbe = &DiagnosticWeaponProbeFile{} },
		func(f *PPOFile) { h := *f.DiagnosticAnchorProbe; h.Mode = "force"; f.DiagnosticAnchorProbe = &h },
		func(f *PPOFile) { h := *f.DiagnosticAnchorProbe; h.Frame = 0; f.DiagnosticAnchorProbe = &h },
	} {
		g := f
		change(&g)
		if validateDiagnosticAnchorProbe(g) == nil {
			t.Fatal("invalid header accepted")
		}
	}
}
