package policy

import "fmt"

const DiagnosticWideGateVersion = "combat_one_frame_wide_gate_clamp_v1"

// Diagnostic acquisition only. The changed gate distribution is sampled before
// ordinary conditional fire. Its density is never ordinary-policy PPO data.
type DiagnosticWideGateFile struct {
	Version string `json:"version"`
	Frame   int    `json:"frame"`
	Choice  int    `json:"choice"`
}

func validateDiagnosticWideGate(f PPOFile) error {
	h := f.DiagnosticWideGate
	if h == nil {
		return nil
	}
	if h.Version != DiagnosticWideGateVersion || h.Frame <= 0 || h.Frame > 2147483647 ||
		(h.Choice != 0 && h.Choice != 1) || f.WideAim == nil || !f.WideAim.Enabled ||
		f.ActionDistribution != WideAimDistribution || f.Deterministic || f.ExecutionMode != "" ||
		f.DiagnosticAnchorProbe != nil || f.DiagnosticAimProbe != nil || f.DiagnosticMovementProbe != nil ||
		f.DiagnosticWeaponProbe != nil {
		return fmt.Errorf("invalid diagnostic wide gate clamp")
	}
	if f.DiagnosticContinuation != nil {
		return validateDiagnosticContinuation(f)
	}
	return nil
}

func (p *PPO) diagnosticWideGateLogit(frame int) (float64, bool) {
	h := p.file.DiagnosticWideGate
	if h == nil || frame != h.Frame {
		return 0, false
	}
	if h.Choice == 1 {
		return 20, true
	}
	return -20, true
}
