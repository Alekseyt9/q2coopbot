package policy

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
)

// Probe is a bounded local diagnostic provider, not a trained policy or teacher.
// It exercises direct control using a repeating sequence indexed by game frames.
type Probe struct {
	version string
	steps   []Action
	start   Identity
}

func LoadProbe(path string) (*Probe, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) > 65536 {
		return nil, fmt.Errorf("probe file too large")
	}
	var file struct {
		Kind  string   `json:"kind"`
		Steps []Action `json:"steps"`
	}
	if err := json.Unmarshal(data, &file); err != nil {
		return nil, err
	}
	if file.Kind != "combat_control_probe_v1" || len(file.Steps) == 0 || len(file.Steps) > 256 {
		return nil, fmt.Errorf("expected bounded combat_control_probe_v1")
	}
	o := Observation{Version: ObservationVersion, Health: 100, Identity: Identity{Frame: 1}}
	for _, step := range file.Steps {
		step.Version, step.Identity = ActionVersion, o.Identity
		if step.Weapon != "" {
			return nil, fmt.Errorf("diagnostic probe keeps the current weapon")
		}
		if _, err := Command(o, step, [3]int16{}); err != nil {
			return nil, err
		}
	}
	hash := sha256.Sum256(data)
	return &Probe{version: "diagnostic_probe:" + hex.EncodeToString(hash[:]), steps: file.Steps}, nil
}

func (p *Probe) Version() string { return p.version }

// NewSession preserves the loaded bytes/version but resets per-life progress.
func (p *Probe) NewSession() *Probe {
	return &Probe{version: p.version, steps: append([]Action(nil), p.steps...)}
}

func (p *Probe) Decide(o Observation) (Action, error) {
	id := o.Identity
	if !SameLife(p.start, id) || id.Frame < p.start.Frame || p.start.Frame == 0 {
		p.start = id
	}
	a := p.steps[(id.Frame-p.start.Frame)%len(p.steps)]
	a.Version, a.Identity = ActionVersion, id
	return a, nil
}

func SameLife(a, b Identity) bool {
	a.Frame, b.Frame = 0, 0
	return a == b
}
