package policy

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
)

const HJEPASamplingCheckpointVersion = "hjepa_sampling_checkpoint_v1"
const HJEPASamplingBranchCheckpointVersion = "hjepa_sampling_checkpoint_v2"

// HJEPASamplingCheckpoint preserves the logical draw identity across native
// load/reconnect. Actions and observations keep their actual native identity.
// This is a policy RNG checkpoint, not proof of a restored native world.
type HJEPASamplingCheckpoint struct {
	Stream  int64    `json:"sampling_stream,omitempty"`
	Version string   `json:"version"`
	Model   string   `json:"model"`
	Source  Identity `json:"source_identity"`
}

type hSamplingRestore struct {
	source, native Identity
	stream         int64
}

func (p *HJEPA) SamplingStream() int64 {
	if p.samplingRestore == nil {
		return 0
	}
	return p.samplingRestore.stream
}

// A fragment stream changes only draw coordinates, never actor parameters or
// the action density. Sample.SamplingSeed remains the frozen model's base seed;
// exact replay also needs this stream and the logical checkpoint identity.
func (p *HJEPA) samplingSeed(base int64) int64 {
	stream := p.SamplingStream()
	if stream == 0 {
		return base
	}
	key, _ := json.Marshal(struct {
		Domain       string
		Seed, Stream int64
	}{"hjepa_fragment_sampling_seed_v1", base, stream})
	h := sha256.Sum256(key)
	return int64(binary.LittleEndian.Uint64(h[:8]) & ((uint64(1) << 63) - 1))
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
	version := HJEPASamplingCheckpointVersion
	if p.SamplingStream() != 0 {
		version = HJEPASamplingBranchCheckpointVersion
	}
	return HJEPASamplingCheckpoint{Version: version, Model: p.version, Source: source, Stream: p.SamplingStream()}, nil
}

// RestoreSamplingCheckpoint binds a fresh policy to the first native load
// observation. It preserves both primitive and categorical weapon draws.
func (p *HJEPA) RestoreSamplingCheckpoint(state HJEPASamplingCheckpoint, native Identity) error {
	validVersion := state.Version == HJEPASamplingCheckpointVersion && state.Stream == 0 || state.Version == HJEPASamplingBranchCheckpointVersion
	if !validVersion || state.Stream < 0 || state.Stream > 2147483647 || state.Model != p.version || !validSamplingAnchor(state.Source) ||
		!validSamplingAnchor(native) || state.Source.Map != native.Map || state.Source.Actor != native.Actor ||
		p.file.Planner != "bc" || p.file.PrimitiveSampling == nil || p.file.WeaponOptionFrames != 0 ||
		p.samplingRestore != nil || p.lastSample != nil || p.weaponOption.valid {
		return fmt.Errorf("H-JEPA sampling checkpoint/model/native anchor rejected")
	}
	p.samplingRestore = &hSamplingRestore{source: state.Source, native: native, stream: state.Stream}
	return nil
}
