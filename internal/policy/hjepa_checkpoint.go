package policy

import "fmt"

const HJEPASamplingCheckpointVersion = "hjepa_sampling_checkpoint_v1"

// HJEPASamplingCheckpoint preserves the logical draw identity across native
// load/reconnect. Actions and observations keep their actual native identity.
// This is a policy RNG checkpoint, not proof of a restored native world.
type HJEPASamplingCheckpoint struct {
	Version string   `json:"version"`
	Model   string   `json:"model"`
	Source  Identity `json:"source_identity"`
}

type hSamplingRestore struct {
	source, native Identity
}

func validSamplingAnchor(id Identity) bool {
	return id.Map != "" && len(id.Map) <= 64 && id.Frame > 0 && id.Frame <= 1000000 &&
		id.Connection > 0 && id.Spawncount > 0 && id.Actor > 0 && id.Life > 0
}

func (p *HJEPA) samplingIdentity(native Identity) (Identity, error) {
	if p.samplingRestore == nil {
		return native, nil
	}
	r := p.samplingRestore
	if !SameLife(r.native, native) || native.Frame < r.native.Frame || native.Frame-r.native.Frame > 1000000-r.source.Frame {
		return Identity{}, fmt.Errorf("H-JEPA restored sampling identity outside checkpoint life/clock")
	}
	id := r.source
	id.Frame += native.Frame - r.native.Frame
	return id, nil
}

func (p *HJEPA) CaptureSamplingCheckpoint(native Identity) (HJEPASamplingCheckpoint, error) {
	if p.file.Planner != "bc" || p.file.PrimitiveSampling == nil || p.file.WeaponOptionFrames != 0 || !validSamplingAnchor(native) {
		return HJEPASamplingCheckpoint{}, fmt.Errorf("H-JEPA checkpoint requires frame-wise sampled BC actor and valid anchor")
	}
	source, err := p.samplingIdentity(native)
	if err != nil {
		return HJEPASamplingCheckpoint{}, err
	}
	return HJEPASamplingCheckpoint{Version: HJEPASamplingCheckpointVersion, Model: p.version, Source: source}, nil
}

// RestoreSamplingCheckpoint binds a fresh policy to the first native load
// observation. It preserves both primitive and categorical weapon draws.
func (p *HJEPA) RestoreSamplingCheckpoint(state HJEPASamplingCheckpoint, native Identity) error {
	if state.Version != HJEPASamplingCheckpointVersion || state.Model != p.version || !validSamplingAnchor(state.Source) ||
		!validSamplingAnchor(native) || state.Source.Map != native.Map || state.Source.Actor != native.Actor ||
		p.file.Planner != "bc" || p.file.PrimitiveSampling == nil || p.file.WeaponOptionFrames != 0 ||
		p.samplingRestore != nil || p.lastSample != nil || p.weaponOption.valid {
		return fmt.Errorf("H-JEPA sampling checkpoint/model/native anchor rejected")
	}
	p.samplingRestore = &hSamplingRestore{source: state.Source, native: native}
	return nil
}
