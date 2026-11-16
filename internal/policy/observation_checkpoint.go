package policy

import (
	"encoding/json"
	"fmt"
	"sort"

	"q2coopbot/internal/quake"
)

const ObservationMemoryCheckpointVersion = "combat_observation_memory_checkpoint_v1"

type ThreatMemoryCheckpointRecord struct {
	Enemy    Enemy      `json:"enemy"`
	Position quake.Vec3 `json:"last_observed_position"`
	Frame    int        `json:"last_observed_frame"`
}

// ObservationMemoryCheckpoint contains past client observations only. Native
// player/entity/projectile state must come from the independently verified save.
type ObservationMemoryCheckpoint struct {
	Version   string                         `json:"version"`
	Anchor    Identity                       `json:"anchor"`
	Frames    []HistoryFrame                 `json:"frames"`
	NextTrack int                            `json:"next_track"`
	Threats   []ThreatMemoryCheckpointRecord `json:"threats"`
}

func (s ObservationMemoryCheckpoint) validate() error {
	if s.Version != ObservationMemoryCheckpointVersion || !validSamplingAnchor(s.Anchor) ||
		len(s.Frames) < 1 || len(s.Frames) > HistoryFrames || s.NextTrack < 0 || s.NextTrack > 64000000 || len(s.Threats) > 8 {
		return fmt.Errorf("observation memory checkpoint header/size rejected")
	}
	for i, f := range s.Frames {
		if !SameLife(f.Identity, s.Anchor) || f.Identity.Frame != s.Anchor.Frame-len(s.Frames)+1+i || f.Health <= 0 || len(f.Enemies) > 64 || f.Projectiles != nil && len(*f.Projectiles) > 64 {
			return fmt.Errorf("observation checkpoint history is not a contiguous living prefix")
		}
		groups := [][]Enemy{f.Enemies}
		if f.Projectiles != nil {
			groups = append(groups, *f.Projectiles)
		}
		for _, enemies := range groups {
			for _, e := range enemies {
				if e.ID <= 0 || e.Track == nil || *e.Track < 1 || *e.Track > s.NextTrack {
					return fmt.Errorf("observation checkpoint track counter mismatch")
				}
			}
		}
	}
	seen := map[int]bool{}
	for _, r := range s.Threats {
		if r.Enemy.ID <= 0 || seen[r.Enemy.ID] || r.Frame > s.Anchor.Frame || s.Anchor.Frame-r.Frame > ThreatMemoryFrames {
			return fmt.Errorf("observation checkpoint invalid/stale threat")
		}
		seen[r.Enemy.ID] = true
	}
	if _, err := json.Marshal(s); err != nil {
		return fmt.Errorf("nonfinite observation memory: %w", err)
	}
	return nil
}

func copyObservationMemory(s ObservationMemoryCheckpoint) (ObservationMemoryCheckpoint, error) {
	data, err := json.Marshal(s)
	if err != nil {
		return ObservationMemoryCheckpoint{}, err
	}
	var result ObservationMemoryCheckpoint
	err = json.Unmarshal(data, &result)
	return result, err
}

func CaptureObservationMemory(history *History, threats *ThreatMemory, anchor Identity) (ObservationMemoryCheckpoint, error) {
	if history == nil || threats == nil || threats.identity != anchor {
		return ObservationMemoryCheckpoint{}, fmt.Errorf("observation memory anchor mismatch")
	}
	s := ObservationMemoryCheckpoint{Version: ObservationMemoryCheckpointVersion, Anchor: anchor, Frames: history.frames, NextTrack: history.nextTrack}
	for _, r := range threats.known {
		s.Threats = append(s.Threats, ThreatMemoryCheckpointRecord{Enemy: r.enemy, Position: r.position, Frame: r.frame})
	}
	sort.Slice(s.Threats, func(i, j int) bool { return s.Threats[i].Enemy.ID < s.Threats[j].Enemy.ID })
	if err := s.validate(); err != nil {
		return ObservationMemoryCheckpoint{}, err
	}
	return copyObservationMemory(s)
}

// RestoreObservationMemory rebases ages, not observations. Old frames can
// become negative when a native load restarts its network clock at frame1.
// Neither current HP nor current hidden monster positions are synthesized.
func RestoreObservationMemory(state ObservationMemoryCheckpoint, native Identity) (History, ThreatMemory, error) {
	return restoreObservationMemory(state, native, false)
}

// RestoreObservationMemoryBeforeDecision restores the prefix immediately before
// root. Frame zero is an internal past-memory anchor when a loaded world starts
// at frame one; it is never a valid observation or action identity.
func RestoreObservationMemoryBeforeDecision(state ObservationMemoryCheckpoint, root Identity) (History, ThreatMemory, error) {
	if !validSamplingAnchor(root) {
		return History{}, ThreatMemory{}, fmt.Errorf("observation memory decision root rejected")
	}
	root.Frame--
	return restoreObservationMemory(state, root, true)
}

func restoreObservationMemory(state ObservationMemoryCheckpoint, native Identity, prior bool) (History, ThreatMemory, error) {
	if err := state.validate(); err != nil {
		return History{}, ThreatMemory{}, err
	}
	valid := validSamplingAnchor(native)
	if prior && native.Frame == 0 {
		check := native
		check.Frame = 1
		valid = validSamplingAnchor(check)
	}
	if !valid || native.Map != state.Anchor.Map || native.Actor != state.Anchor.Actor || native.Life != state.Anchor.Life {
		return History{}, ThreatMemory{}, fmt.Errorf("observation memory native identity rejected")
	}
	s, err := copyObservationMemory(state)
	if err != nil {
		return History{}, ThreatMemory{}, err
	}
	for i := range s.Frames {
		age := state.Anchor.Frame - s.Frames[i].Identity.Frame
		s.Frames[i].Identity = native
		s.Frames[i].Identity.Frame -= age
	}
	h := History{frames: s.Frames, nextTrack: s.NextTrack}
	m := ThreatMemory{identity: native, known: make(map[int]threatRecord)}
	for _, r := range s.Threats {
		m.known[r.Enemy.ID] = threatRecord{enemy: r.Enemy, position: r.Position, frame: native.Frame - (state.Anchor.Frame - r.Frame)}
	}
	return h, m, nil
}
