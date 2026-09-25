package bot

import (
	"math"

	"q2coopbot/internal/quake"
)

// TeammateMotion is a coarse, conditional footprint around the last sighting.
// It does not prove that any area is reachable or that the player stayed inside
// the footprint: an unobserved teleport or unusual movement can invalidate it.
type TeammateMotion struct {
	Entity            int     `json:"entity"`
	AgeFrames         int     `json:"age_frames"`
	Radius            float64 `json:"radius"`
	NearbyGroundAreas int     `json:"nearby_ground_areas"`
	TotalGroundAreas  int     `json:"total_ground_areas"`
	AreaIDs           []int   `json:"area_ids,omitempty"`
	Method            string  `json:"method"`
}

const (
	ordinaryMotionPerFrame = 40.0 // diagnostic 400 units/s at 10 game frames/s
	motionOriginMargin     = 32.0
	maxMotionAreaIDs       = 64
)

func (p *Planner) updateTeammateMotion(s quake.Snapshot) {
	p.World.TeammateMotion = nil
	if p.Nav == nil || s.Teammate != nil || s.LastTeammate == nil || s.LastTeammateEntity <= 0 ||
		s.TeammateAgeFrames == nil || *s.TeammateAgeFrames <= 0 || *s.TeammateAgeFrames > 40 {
		return
	}
	age := *s.TeammateAgeFrames
	radius := motionOriginMargin + ordinaryMotionPerFrame*float64(age)
	motion := &TeammateMotion{Entity: s.LastTeammateEntity, AgeFrames: age, Radius: radius,
		Method: "conditional_horizontal_radius"}
	for id := 1; id < len(p.Nav.Areas); id++ {
		area := p.Nav.Areas[id]
		if area.Flags&1 == 0 {
			continue
		}
		motion.TotalGroundAreas++
		dx := math.Max(area.Min[0]-(*s.LastTeammate)[0], math.Max(0, (*s.LastTeammate)[0]-area.Max[0]))
		dy := math.Max(area.Min[1]-(*s.LastTeammate)[1], math.Max(0, (*s.LastTeammate)[1]-area.Max[1]))
		if math.Hypot(dx, dy) > radius {
			continue
		}
		motion.NearbyGroundAreas++
		if motion.NearbyGroundAreas <= maxMotionAreaIDs {
			motion.AreaIDs = append(motion.AreaIDs, id)
		}
	}
	// A truncated list could be mistaken for the whole envelope. Keep the count
	// but omit the IDs when the set is too broad to report compactly.
	if motion.NearbyGroundAreas > maxMotionAreaIDs {
		motion.AreaIDs = nil
	}
	p.World.TeammateMotion = motion
}
