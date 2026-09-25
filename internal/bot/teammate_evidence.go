package bot

import "q2coopbot/internal/quake"

// TeammateEvidence separates repeated activity from location observations.
// Counts are scoped to one last-seen player identity and one visual sighting.
type TeammateEvidence struct {
	Entity               int     `json:"entity"`
	LocationStatus       string  `json:"location_status"`
	ActivitySounds       int     `json:"activity_sounds,omitempty"`
	ActivityFrames       int     `json:"activity_frames,omitempty"`
	LastSoundFrame       int     `json:"last_sound_frame,omitempty"`
	LastSoundName        string  `json:"last_sound_name,omitempty"`
	SourceOutsideNominal bool    `json:"source_outside_nominal,omitempty"`
	SourceConflictFrame  int     `json:"source_conflict_frame,omitempty"`
	Reacquired           bool    `json:"reacquired,omitempty"`
	VisualOutsideNominal bool    `json:"visual_outside_nominal,omitempty"`
	ObservedDistance     float64 `json:"observed_distance,omitempty"`
	NominalRadius        float64 `json:"nominal_radius,omitempty"`
	lastSeenFrame        int
	lastProcessedFrame   int
}

func nominalMotionRadius(age int) float64 {
	return motionOriginMargin + ordinaryMotionPerFrame*float64(age)
}

func (p *Planner) updateTeammateEvidence(previous, s quake.Snapshot) {
	p.World.TeammateEvidence = nil
	if s.Teammate != nil {
		if previous.Map == s.Map && previous.Teammate == nil && previous.LastTeammate != nil &&
			previous.LastTeammateEntity == s.TeammateEntity && s.TeammateEntity > 0 &&
			previous.TeammateAgeFrames != nil && *previous.TeammateAgeFrames > 0 &&
			*previous.TeammateAgeFrames <= 40 && s.Frame >= previous.Frame {
			age := s.Frame - (previous.Frame - *previous.TeammateAgeFrames)
			if age <= 40 {
				observed := quake.Horizontal(*previous.LastTeammate, *s.Teammate)
				radius := nominalMotionRadius(age)
				event := &TeammateEvidence{Entity: s.TeammateEntity, LocationStatus: "observed",
					Reacquired: true, VisualOutsideNominal: observed > radius,
					ObservedDistance: observed, NominalRadius: radius}
				if p.teammateEvidence != nil && p.teammateEvidence.Entity == s.TeammateEntity &&
					p.teammateEvidence.lastSeenFrame == previous.Frame-*previous.TeammateAgeFrames {
					event.ActivitySounds = p.teammateEvidence.ActivitySounds
					event.ActivityFrames = p.teammateEvidence.ActivityFrames
				}
				p.World.TeammateEvidence = event
			}
		}
		p.teammateEvidence = nil
		return
	}
	if s.LastTeammate == nil || s.LastTeammateEntity <= 0 || s.TeammateAgeFrames == nil ||
		*s.TeammateAgeFrames <= 0 || *s.TeammateAgeFrames > 40 {
		p.teammateEvidence = nil
		return
	}
	lastSeenFrame := s.Frame - *s.TeammateAgeFrames
	if p.teammateEvidence == nil || p.teammateEvidence.Entity != s.LastTeammateEntity ||
		p.teammateEvidence.lastSeenFrame != lastSeenFrame || s.Frame < p.teammateEvidence.lastProcessedFrame {
		p.teammateEvidence = &TeammateEvidence{Entity: s.LastTeammateEntity, LocationStatus: "unknown",
			lastSeenFrame: lastSeenFrame, lastProcessedFrame: lastSeenFrame}
	}
	state := p.teammateEvidence
	if s.Frame > state.lastProcessedFrame {
		soundsThisFrame := 0
		for _, sound := range s.Sounds {
			if sound.Entity != s.LastTeammateEntity || sound.Name == "" || sound.Attenuation <= 0 {
				continue
			}
			state.ActivitySounds++
			soundsThisFrame++
			state.LastSoundFrame, state.LastSoundName = s.Frame, sound.Name
			// SND_POS is the sound source, which need not be the player's origin.
			if sound.Position != nil && quake.Horizontal(*s.LastTeammate, *sound.Position) > nominalMotionRadius(*s.TeammateAgeFrames) {
				state.SourceOutsideNominal = true
				state.SourceConflictFrame = s.Frame
			}
		}
		if soundsThisFrame > 0 {
			state.ActivityFrames++
		}
		state.lastProcessedFrame = s.Frame
	}
	if state.ActivitySounds > 0 {
		copy := *state
		p.World.TeammateEvidence = &copy
	}
}
