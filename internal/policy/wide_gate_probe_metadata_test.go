package policy

import "testing"

func TestWideGateProbeMetadataIsolation(t *testing.T) {
	for _, choice := range []int{0, 1} {
		h := &DiagnosticWideGateFile{Version: DiagnosticWideGateVersion, Frame: 123, Choice: choice}
		f := PPOFile{ActionDistribution: WideAimDistribution, WideAim: &WideAimFile{Enabled: true}, DiagnosticWideGate: h}
		if err := validateDiagnosticWideGate(f); err != nil {
			t.Fatal(err)
		}
		p := &PPO{file: f}
		if !p.IsDiagnosticCombatProbe() || p.IsStochastic() {
			t.Fatal("diagnostic admitted as ordinary stochastic policy")
		}
		for _, frame := range []int{122, 123, 124} {
			logit, active := p.diagnosticWideGateLogit(frame)
			if active != (frame == 123) {
				t.Fatal("clamp leaked outside root")
			}
			if active && logit != float64(40*choice-20) {
				t.Fatal("wrong clamp")
			}
		}
		for _, mutate := range []func(*PPOFile){
			func(f *PPOFile) { f.DiagnosticWideGate.Frame = 0 },
			func(f *PPOFile) { f.DiagnosticWideGate.Choice = 2 },
			func(f *PPOFile) { f.DiagnosticWideGate.Version = "bad" },
			func(f *PPOFile) { f.WideAim = nil },
			func(f *PPOFile) { f.Deterministic = true },
			func(f *PPOFile) { f.DiagnosticAnchorProbe = &DiagnosticAnchorProbeFile{} },
		} {
			bad := f
			copyHeader := *h
			bad.DiagnosticWideGate = &copyHeader
			mutate(&bad)
			if validateDiagnosticWideGate(bad) == nil {
				t.Fatal("invalid diagnostic header accepted")
			}
		}
	}
	p := &PPO{}
	if p.IsDiagnosticCombatProbe() {
		t.Fatal("ordinary policy changed")
	}
	if _, active := p.diagnosticWideGateLogit(123); active {
		t.Fatal("ordinary clamp")
	}
}
