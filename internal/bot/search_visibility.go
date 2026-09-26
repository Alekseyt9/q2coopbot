package bot

import (
	"math"
	"sort"

	"q2coopbot/internal/quake"
)

// Static BSP rays estimate information gain, not server PVS or player presence.
// Dynamic doors can occlude these rays; movement keeps its separate door guard.
type SearchVisibility struct {
	HiddenSamples   int  `json:"hidden_samples"`
	NewlyVisible    int  `json:"newly_visible"`
	SafeCandidates  int  `json:"safe_candidates"`
	MaxNewlyVisible int  `json:"max_newly_visible"`
	AuditComplete   bool `json:"audit_complete"`
	AuditSamples    int  `json:"audit_samples"`
	AuditMaxGain    int  `json:"audit_max_gain"`
	AuditChosenGain int  `json:"audit_chosen_gain"`
}

func (v *SearchVisibility) noNewCoverage() bool {
	return v != nil && v.HiddenSamples > 0 && v.SafeCandidates > 0 && v.MaxNewlyVisible == 0 &&
		v.AuditComplete && v.AuditSamples > 0 && v.AuditMaxGain == 0
}

// A bounded second pass protects against centre-only false negatives.
// An incomplete pass cannot authorize rejection; the sparse pass remains
// useful for ranking but is not evidence that all nearby space was examined.
func (p *Planner) searchVisibilityAuditSamples(s quake.Snapshot) ([]quake.Vec3, bool) {
	var samples []quake.Vec3
	seen := map[quake.Vec3]bool{}
	last := *s.LastTeammate
	for _, area := range p.Nav.Areas[1:] {
		if area.Flags&1 == 0 {
			continue
		}
		for _, d := range []quake.Vec3{{}, {16, 0, 0}, {-16, 0, 0}, {0, 16, 0}, {0, -16, 0}} {
			point := quake.Vec3{area.Center[0] + d[0], area.Center[1] + d[1], area.Min[2] + 1}
			if seen[point] || quake.Horizontal(last, point) > 320 || math.Abs(last[2]-point[2]) > 48 {
				continue
			}
			seen[point] = true
			if !p.World.Geometry.PlayerMoveClear(point, point) {
				continue
			}
			if _, ok := p.World.Geometry.GroundDrop(point, 24); !ok {
				continue
			}
			if !unseenSearchSample(point, s.Self, p.lastSeenSelf, p.lastSeenSelfKnown, p.World.Geometry.ClearShot) {
				continue
			}
			if len(samples) == 128 {
				return samples, false
			}
			samples = append(samples, point)
		}
	}
	return samples, true
}

func searchEye(origin quake.Vec3) quake.Vec3 {
	origin[2] += 22
	return origin
}

func searchViewpointScore(travel, fromLast float64, gain int) float64 {
	// Bound the detour preference to 128 units; zero gain preserves the old
	// distance ordering. All candidates still satisfy the 320-unit route cap.
	return travel + fromLast*0.5 - 16*float64(min(gain, 8))
}

func newVisibleSamples(samples []quake.Vec3, candidate quake.Vec3, clear func(quake.Vec3, quake.Vec3) bool) int {
	count := 0
	for _, sample := range samples {
		if clear(searchEye(candidate), searchEye(sample)) {
			count++
		}
	}
	return count
}

func unseenSearchSample(point, current, previous quake.Vec3, previousKnown bool, clear func(quake.Vec3, quake.Vec3) bool) bool {
	return !clear(searchEye(current), searchEye(point)) &&
		(!previousKnown || !clear(searchEye(previous), searchEye(point)))
}

func (p *Planner) searchVisibilitySamples(s quake.Snapshot) []quake.Vec3 {
	if p.Nav == nil || len(p.Nav.Areas) < 2 || p.World.Geometry == nil || s.LastTeammate == nil {
		return nil
	}
	var candidates []quake.Vec3
	last := *s.LastTeammate
	for _, area := range p.Nav.Areas[1:] {
		point := area.Center
		point[2] = area.Min[2] + 1
		if area.Flags&1 == 0 || quake.Horizontal(last, point) > 320 || math.Abs(last[2]-point[2]) > 48 {
			continue
		}
		candidates = append(candidates, point)
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		return quake.Horizontal(last, candidates[i]) < quake.Horizontal(last, candidates[j])
	})
	// Spatial separation avoids rewarding many tiny AAS areas in one place.
	var samples []quake.Vec3
	for _, point := range candidates {
		near := false
		for _, existing := range samples {
			if quake.Horizontal(point, existing) < 32 {
				near = true
				break
			}
		}
		if near || !p.World.Geometry.PlayerMoveClear(point, point) {
			continue
		}
		if _, ok := p.World.Geometry.GroundDrop(point, 24); !ok {
			continue
		}
		if !unseenSearchSample(point, s.Self, p.lastSeenSelf, p.lastSeenSelfKnown, p.World.Geometry.ClearShot) {
			continue
		}
		samples = append(samples, point)
		if len(samples) == 32 {
			break
		}
	}
	return samples
}
