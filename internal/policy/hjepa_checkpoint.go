package policy

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
)

const HJEPASamplingCheckpointVersion = "hjepa_sampling_checkpoint_v1"
const HJEPASamplingBranchCheckpointVersion = "hjepa_sampling_checkpoint_v2"
const HJEPASamplingFireCheckpointVersion = "hjepa_sampling_checkpoint_v3"
const HJEPASamplingFireOnlyScope = "fire_only_v1"
const HJEPASamplingRootFireCheckpointVersion = "hjepa_sampling_checkpoint_v4"
const HJEPASamplingRootFireScope = "root_fire_only_v1"
const HJEPASamplingRootJointCheckpointVersion = "hjepa_sampling_checkpoint_v5"
const HJEPASamplingRootJointScope = "root_joint_only_v1"

// HJEPASamplingCheckpoint preserves the logical draw identity across native
// load/reconnect. Actions and observations keep their actual native identity.
// This is a policy RNG checkpoint, not proof of a restored native world.
type HJEPASamplingCheckpoint struct {
	Scope   string   `json:"sampling_scope,omitempty"`
	Stream  int64    `json:"sampling_stream,omitempty"`
	Version string   `json:"version"`
	Model   string   `json:"model"`
	Source  Identity `json:"source_identity"`
}

type hSamplingRestore struct {
	source, native Identity
	stream         int64
	scope          string
}

func (p *HJEPA) SamplingScope() string {
	if p.samplingRestore == nil {
		return ""
	}
	return p.samplingRestore.scope
}

func (s HJEPASamplingCheckpoint) Validate() error {
	version := (s.Version == HJEPASamplingCheckpointVersion && s.Stream == 0 || s.Version == HJEPASamplingBranchCheckpointVersion) && s.Scope == "" ||
		s.Version == HJEPASamplingFireCheckpointVersion && s.Scope == HJEPASamplingFireOnlyScope ||
		s.Version == HJEPASamplingRootFireCheckpointVersion && s.Scope == HJEPASamplingRootFireScope ||
		s.Version == HJEPASamplingRootJointCheckpointVersion && s.Scope == HJEPASamplingRootJointScope
	if !version || s.Stream < 0 || s.Stream > 2147483647 || s.Model == "" || !validSamplingAnchor(s.Source) {
		return fmt.Errorf("invalid H-JEPA sampling checkpoint scope/version/anchor")
	}
	return nil
}

// WithStream changes declared draw coordinates without changing model or root.
func (s HJEPASamplingCheckpoint) WithStream(stream int64, scope string) (HJEPASamplingCheckpoint, error) {
	if err := s.Validate(); err != nil {
		return HJEPASamplingCheckpoint{}, err
	}
	s.Stream, s.Scope, s.Version = stream, scope, HJEPASamplingBranchCheckpointVersion
	if scope == HJEPASamplingFireOnlyScope {
		s.Version = HJEPASamplingFireCheckpointVersion
	} else if scope == HJEPASamplingRootFireScope {
		s.Version = HJEPASamplingRootFireCheckpointVersion
	} else if scope == HJEPASamplingRootJointScope {
		s.Version = HJEPASamplingRootJointCheckpointVersion
	}
	return s, s.Validate()
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
func (p *HJEPA) samplingSeed(base int64, native Identity) int64 {
	if p.SamplingScope() == HJEPASamplingFireOnlyScope || p.SamplingScope() == HJEPASamplingRootFireScope {
		return base
	}
	if p.SamplingScope() == HJEPASamplingRootJointScope && native.Frame != p.samplingRestore.native.Frame {
		return base
	}
	return hFragmentSamplingSeed(base, p.SamplingStream())
}

func (p *HJEPA) fireSamplingSeed(base int64, native Identity) int64 {
	if (p.SamplingScope() == HJEPASamplingRootFireScope || p.SamplingScope() == HJEPASamplingRootJointScope) && native.Frame != p.samplingRestore.native.Frame {
		return base
	}
	return hFragmentSamplingSeed(base, p.SamplingStream())
}

func hFragmentSamplingSeed(base, stream int64) int64 {
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
	version, scope, stream := HJEPASamplingCheckpointVersion, p.SamplingScope(), p.SamplingStream()
	// A root-only override has no future effect after its first decision.
	// A nested pre-decision save must not reapply that past intervention.
	if (scope == HJEPASamplingRootFireScope || scope == HJEPASamplingRootJointScope) && native.Frame > p.samplingRestore.native.Frame {
		scope, stream = "", 0
	}
	if stream != 0 {
		version = HJEPASamplingBranchCheckpointVersion
	}
	if scope == HJEPASamplingFireOnlyScope {
		version = HJEPASamplingFireCheckpointVersion
	} else if scope == HJEPASamplingRootFireScope {
		version = HJEPASamplingRootFireCheckpointVersion
	} else if scope == HJEPASamplingRootJointScope {
		version = HJEPASamplingRootJointCheckpointVersion
	}
	return HJEPASamplingCheckpoint{Version: version, Scope: scope, Model: p.version, Source: source, Stream: stream}, nil
}

// RestoreSamplingCheckpoint binds a fresh policy to the first native load
// observation. It preserves both primitive and categorical weapon draws.
func (p *HJEPA) RestoreSamplingCheckpoint(state HJEPASamplingCheckpoint, native Identity) error {
	if state.Validate() != nil || state.Model != p.version ||
		!validSamplingAnchor(native) || state.Source.Map != native.Map || state.Source.Actor != native.Actor ||
		p.file.Planner != "bc" || p.file.PrimitiveSampling == nil || p.file.WeaponOptionFrames != 0 ||
		p.samplingRestore != nil || p.lastSample != nil || p.weaponOption.valid {
		return fmt.Errorf("H-JEPA sampling checkpoint/model/native anchor rejected")
	}
	p.samplingRestore = &hSamplingRestore{source: state.Source, native: native, stream: state.Stream, scope: state.Scope}
	return nil
}
