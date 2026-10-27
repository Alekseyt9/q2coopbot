package policy

import "testing"

// Admission only; no network construction or neural forward.
func TestWideContinuationCompositionMetadata(t *testing.T) {
	base := PPOFile{Features: MemoryRouteFeatureVersion, ActionDistribution: WideAimDistribution,
		WideAim: &WideAimFile{Enabled: true}, AimCandidateHead: &AimCandidateFile{},
		DiagnosticContinuation: &DiagnosticContinuationFile{Version: DiagnosticContinuationVersion, Frame: 124, Seed: 170001}}
	if err := validateDiagnosticContinuation(base); err != nil {
		t.Fatal("fork-only wide rejected", err)
	}
	base.DiagnosticWideGate = &DiagnosticWideGateFile{Version: DiagnosticWideGateVersion, Frame: 123, Choice: 1}
	if err := validateDiagnosticWideGate(base); err != nil {
		t.Fatal("composed clamp rejected", err)
	}
	for _, change := range []func(*PPOFile){
		func(f *PPOFile) { f.DiagnosticContinuation.Frame = 123 },
		func(f *PPOFile) { f.DiagnosticContinuation.Frame = 125 },
		func(f *PPOFile) { f.Features = "unknown" },
		func(f *PPOFile) { f.WideAim.Enabled = false },
		func(f *PPOFile) { f.DiagnosticMovementProbe = &DiagnosticMovementProbeFile{} },
		func(f *PPOFile) { f.DiagnosticAnchorProbe = &DiagnosticAnchorProbeFile{} },
	} {
		f := base
		fork, head := *base.DiagnosticContinuation, *base.WideAim
		f.DiagnosticContinuation, f.WideAim = &fork, &head
		change(&f)
		if validateDiagnosticContinuation(f) == nil {
			t.Fatal("invalid wide continuation admitted")
		}
	}
	p := PPO{file: base}
	if !p.IsDiagnosticCombatProbe() || p.IsStochastic() {
		t.Fatal("composed fork admitted as ordinary PPO")
	}
}
