package policy

import "testing"

// Metadata only: no model inference, sampling or neural arithmetic on CPU.
func TestDiagnosticAimProbeMetadata(t *testing.T) {
	f := PPOFile{
		AimCandidateHead: &AimCandidateFile{Layers: []DenseLayer{{}, {
			Weight: [][]float64{make([]float64, 64)}, Bias: make([]float64, 25),
		}}},
		DiagnosticAimProbe: &DiagnosticAimProbeFile{
			Version: DiagnosticAimProbeVersion, Frame: 110, Logits: make([]float64, 25),
		},
	}
	if err := validateDiagnosticAimProbe(f); err != nil {
		t.Fatal(err)
	}
	if (&PPO{file: f}).IsStochastic() {
		t.Fatal("diagnostic acquisition advertised as ordinary PPO eligible")
	}
	f.AimCandidateHead.Layers[1].Weight[0][0] = 1
	if validateDiagnosticAimProbe(f) == nil {
		t.Fatal("nonconstant baseline accepted")
	}
	f.DiagnosticAimProbe.Version = DiagnosticPolicyAimProbeVersion
	if err := validateDiagnosticAimProbe(f); err != nil {
		t.Fatal("ordinary-policy V2 probe rejected:", err)
	}
	if (&PPO{file: f}).IsStochastic() {
		t.Fatal("V2 diagnostic acquisition advertised as PPO eligible")
	}
	f.DiagnosticAimProbe.Version = "unknown"
	if validateDiagnosticAimProbe(f) == nil {
		t.Fatal("unknown diagnostic version accepted")
	}
	f.DiagnosticAimProbe.Version = DiagnosticAimProbeVersion
	f.AimCandidateHead.Layers[1].Weight[0][0] = 0
	f.DiagnosticAimProbe.Logits[0] = -61
	if validateDiagnosticAimProbe(f) == nil {
		t.Fatal("logit span with unsafe support accepted")
	}
	f.DiagnosticAimProbe.Logits[0] = 0
	f.DiagnosticAimProbe.Frame = 0
	if validateDiagnosticAimProbe(f) == nil {
		t.Fatal("nonpositive frame accepted")
	}
}
