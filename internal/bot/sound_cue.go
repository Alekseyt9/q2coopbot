package bot

import (
	"sort"

	"q2coopbot/internal/quake"
)

// TeammateSoundCue records activity of a previously seen player, not a location.
type TeammateSoundCue struct {
	Entity    int    `json:"entity"`
	Name      string `json:"name"`
	Frame     int    `json:"frame"`
	AgeFrames int    `json:"age_frames"`
	// SourceAreas are sound-source hypotheses only when this packet carries SND_POS.
	// The explicit sound origin is not proof of the player's current location.
	SourceAreas []int `json:"source_areas,omitempty"`
	// PHS counts are conditional diagnostics, not a hard location constraint.
	PHSPossibleClusters int `json:"phs_possible_clusters,omitempty"`
	PHSTotalClusters    int `json:"phs_total_clusters,omitempty"`
}

const teammateSoundCueFrames = 10

func (p *Planner) updateTeammateSoundCue(s quake.Snapshot) {
	p.World.TeammateSound = nil
	if s.Teammate != nil || s.LastTeammate == nil || s.LastTeammateEntity == 0 ||
		s.TeammateAgeFrames == nil || *s.TeammateAgeFrames <= 0 || *s.TeammateAgeFrames > 40 {
		p.teammateSoundCue = nil
		return
	}
	if p.teammateSoundCue != nil &&
		(p.teammateSoundCue.Entity != s.LastTeammateEntity || s.Frame < p.teammateSoundCue.Frame) {
		p.teammateSoundCue = nil
	}
	for _, sound := range s.Sounds {
		if sound.Entity == s.LastTeammateEntity && sound.Name != "" && sound.Attenuation > 0 {
			cue := &TeammateSoundCue{Entity: sound.Entity, Name: sound.Name, Frame: s.Frame,
				SourceAreas: soundAreaCandidates(p.Nav, sound.Position)}
			if sound.Position == nil && p.World.Geometry != nil {
				if possible, total, ok := p.World.Geometry.PHSPossibleSources(s.Self); ok {
					cue.PHSPossibleClusters, cue.PHSTotalClusters = possible, total
				}
			}
			p.teammateSoundCue = cue
		}
	}
	if p.teammateSoundCue == nil {
		return
	}
	copy := *p.teammateSoundCue
	copy.AgeFrames = s.Frame - copy.Frame
	if copy.AgeFrames > teammateSoundCueFrames {
		p.teammateSoundCue = nil
		return
	}
	p.World.TeammateSound = &copy
}

// An entity-only sound may be multicast across a broad PHS. It cannot narrow
// the player's location. Explicit SND_POS permits a small, untrusted AAS set.
func soundAreaCandidates(nav *quake.Navigator, position *quake.Vec3) []int {
	if nav == nil || position == nil {
		return nil
	}
	type candidate struct {
		id    int
		score float64
	}
	var found []candidate
	for id := 1; id < len(nav.Areas); id++ {
		area := nav.Areas[id]
		if area.Flags&1 == 0 || (*position)[2] < area.Min[2]-16 || (*position)[2] > area.Max[2]+16 ||
			(*position)[0] < area.Min[0]-16 || (*position)[0] > area.Max[0]+16 ||
			(*position)[1] < area.Min[1]-16 || (*position)[1] > area.Max[1]+16 {
			continue
		}
		found = append(found, candidate{id: id, score: quake.Horizontal(*position, area.Center)})
	}
	sort.Slice(found, func(i, j int) bool {
		if found[i].score != found[j].score {
			return found[i].score < found[j].score
		}
		return found[i].id < found[j].id
	})
	if len(found) > 4 {
		found = found[:4]
	}
	ids := make([]int, len(found))
	for i, candidate := range found {
		ids[i] = candidate.id
	}
	return ids
}
