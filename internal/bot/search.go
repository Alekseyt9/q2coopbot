package bot

import (
	"math"

	"q2coopbot/internal/quake"
)

// SearchAttempt records one bounded test of a viewpoint near the last sighting.
// It never asserts that the teammate is at the target.
type SearchAttempt struct {
	Entity              int               `json:"entity"`
	LastSeenFrame       int               `json:"last_seen_frame"`
	Target              *quake.Vec3       `json:"target,omitempty"`
	Basis               string            `json:"basis"`
	ExpectedObservation string            `json:"expected_observation"`
	Attempt             int               `json:"attempt"`
	MaxAttempts         int               `json:"max_attempts"`
	StartFrame          int               `json:"start_frame"`
	EndFrame            int               `json:"end_frame,omitempty"`
	State               string            `json:"state"`
	Outcome             string            `json:"outcome,omitempty"`
	Visibility          *SearchVisibility `json:"visibility,omitempty"`
}

type SearchRouteCheck struct {
	FromArea        int      `json:"from_area"`
	ToArea          int      `json:"to_area"`
	GraphRouteFound bool     `json:"graph_route_found"`
	Travel          *float64 `json:"travel_horizontal_units,omitempty"`
	Limit           float64  `json:"limit_horizontal_units"`
	Reason          string   `json:"reason"`
}

func (p *Planner) finishSearchAttempt(frame int, outcome string) {
	if p.searchAttempt == nil || p.searchAttempt.State != "active" {
		return
	}
	p.searchAttempt.State = "completed"
	p.searchAttempt.Outcome = outcome
	p.searchAttempt.EndFrame = frame
}

// hiddenTeammateGoal permits one short approach to a confirmed position and
// one nearby viewpoint. Neither point is treated as the teammate's position.
func (p *Planner) hiddenTeammateGoal(s quake.Snapshot) (quake.Vec3, string, bool) {
	if p.TestDisableSearch || p.testSetupHold {
		return quake.Vec3{}, "", false
	}
	if p.searchAttempt != nil && s.TeammateAgeFrames != nil &&
		(p.searchAttempt.Entity != s.LastTeammateEntity ||
			p.searchAttempt.LastSeenFrame != s.Frame-*s.TeammateAgeFrames) {
		p.searchAttempt = nil
	}
	if s.LastTeammate == nil || s.TeammateAgeFrames == nil || *s.TeammateAgeFrames <= 0 ||
		*s.TeammateAgeFrames > 40 || p.World.GeometryStatus != "ready" || p.Nav == nil ||
		s.Health <= 0 || !s.OnGround || quake.Horizontal(s.Self, *s.LastTeammate) > 512 ||
		math.Abs(s.Self[2]-(*s.LastTeammate)[2]) > 80 {
		p.finishSearchAttempt(s.Frame, "preconditions_lost")
		return quake.Vec3{}, "", false
	}
	if p.probeTarget != nil {
		if quake.Horizontal(s.Self, p.probeLastSelf) > 8 {
			p.probeLastSelf = s.Self
			p.probeProgressFrame = s.Frame
		}
		if s.Frame-p.probeProgressFrame > 8 || quake.Horizontal(s.Self, *p.probeTarget) <= 16 {
			if s.Frame-p.probeProgressFrame > 8 {
				p.finishSearchAttempt(s.Frame, "stalled")
			} else {
				p.finishSearchAttempt(s.Frame, "not_seen")
			}
			p.probeTarget = nil
			return quake.Vec3{}, "", false
		}
		return *p.probeTarget, "probe_last_seen", true
	}
	if p.probeAttempted {
		return quake.Vec3{}, "", false
	}
	if quake.Horizontal(s.Self, *s.LastTeammate) > 64 {
		p.searchApproachStarted = true
		return *s.LastTeammate, "search_last_seen", true
	}
	if !p.searchApproachStarted || p.TestDisableProbe {
		return quake.Vec3{}, "", false
	}
	if !p.probeAttempted {
		p.probeAttempted = true
		if viewpoint, visibility, ok := p.selectSearchViewpoint(s); ok {
			p.searchAttempt = &SearchAttempt{Entity: s.LastTeammateEntity,
				LastSeenFrame: s.Frame - *s.TeammateAgeFrames, Target: &viewpoint,
				Basis: "last_seen_aas_viewpoint", ExpectedObservation: "teammate_visible_in_current_snapshot",
				Attempt: 1, MaxAttempts: 1, StartFrame: s.Frame, State: "active", Visibility: visibility}
			if visibility.noNewCoverage() {
				p.finishSearchAttempt(s.Frame, "no_new_visibility")
				return quake.Vec3{}, "", false
			}
			p.probeTarget = &viewpoint
			p.probeLastSelf = s.Self
			p.probeProgressFrame = s.Frame
		} else {
			p.searchAttempt = &SearchAttempt{Entity: s.LastTeammateEntity,
				LastSeenFrame: s.Frame - *s.TeammateAgeFrames,
				Basis:         "last_seen_aas_viewpoint", ExpectedObservation: "teammate_visible_in_current_snapshot",
				Attempt: 1, MaxAttempts: 1, StartFrame: s.Frame, EndFrame: s.Frame,
				State: "completed", Outcome: "no_safe_viewpoint"}
		}
	}
	if p.probeTarget != nil && quake.Horizontal(s.Self, *p.probeTarget) > 16 {
		return *p.probeTarget, "probe_last_seen", true
	}
	return quake.Vec3{}, "", false
}

func safeSearchRoute(route []quake.Waypoint, from, goal quake.Vec3, maxTravel float64) (float64, bool) {
	travel, reason := checkSearchRoute(route, from, goal, maxTravel)
	return travel, reason == "ready"
}

func checkSearchRoute(route []quake.Waypoint, from, goal quake.Vec3, maxTravel float64) (float64, string) {
	travel, at := 0.0, from
	reason := "ready"
	for _, waypoint := range route {
		if waypoint.Jump {
			reason = "requires_jump"
		}
		if waypoint.Kind == 11 || waypoint.ElevatorPhase != "" {
			reason = "requires_elevator"
		}
		travel += quake.Horizontal(at, waypoint.Position)
		at = waypoint.Position
	}
	travel += quake.Horizontal(at, goal)
	if reason == "ready" && travel > maxTravel {
		reason = "distance_budget_exceeded"
	}
	return travel, reason
}

func (p *Planner) selectSearchViewpoint(s quake.Snapshot) (quake.Vec3, *SearchVisibility, bool) {
	if p.Nav == nil || p.World.Geometry == nil || s.LastTeammate == nil {
		return quake.Vec3{}, nil, false
	}
	last := *s.LastTeammate
	best := math.Inf(1)
	var chosen quake.Vec3
	var visibility *SearchVisibility
	safeCandidates, maxGain := 0, 0
	samples := p.searchVisibilitySamples(s)
	for i := 1; i < len(p.Nav.Areas); i++ {
		area := p.Nav.Areas[i]
		candidate := area.Center
		candidate[2] = area.Min[2] + 1
		fromLast := quake.Horizontal(last, candidate)
		if area.Flags&1 == 0 || area.Max[0]-area.Min[0] < 24 || area.Max[1]-area.Min[1] < 24 ||
			fromLast < 80 || fromLast > 240 ||
			math.Abs(candidate[2]-last[2]) > 48 || quake.Horizontal(s.Self, candidate) < 48 {
			continue
		}
		if p.lastSeenSelfKnown && quake.Horizontal(p.lastSeenSelf, candidate) < 80 {
			continue
		}
		if _, ok := p.World.Geometry.GroundDrop(candidate, 24); !ok ||
			!p.World.Geometry.PlayerMoveClear(candidate, candidate) {
			continue
		}
		route, ok := p.Nav.SearchRoute(s.Self, candidate)
		if !ok {
			continue
		}
		travel, safe := safeSearchRoute(route, s.Self, candidate, 320)
		if !safe || !p.searchRouteDoorsClear(s.Movers, s.Self, route, candidate) {
			continue
		}
		gain := newVisibleSamples(samples, candidate, p.World.Geometry.ClearShot)
		safeCandidates++
		maxGain = max(maxGain, gain)
		score := searchViewpointScore(travel, fromLast, gain)
		if score < best {
			best, chosen = score, candidate
			visibility = &SearchVisibility{HiddenSamples: len(samples), NewlyVisible: gain}
		}
	}
	if visibility != nil {
		visibility.SafeCandidates, visibility.MaxNewlyVisible = safeCandidates, maxGain
	}
	return chosen, visibility, !math.IsInf(best, 1)
}

// AAS reachability does not encode current door state. Check short segments
// against the observed mover state before committing to a viewpoint.
func (p *Planner) searchRouteDoorsClear(movers []quake.Mover, from quake.Vec3, route []quake.Waypoint, goal quake.Vec3) bool {
	points := make([]quake.Vec3, 0, len(route)+1)
	for _, waypoint := range route {
		points = append(points, waypoint.Position)
	}
	points = append(points, goal)
	at := from
	for _, point := range points {
		distance := quake.Horizontal(at, point)
		if distance < 0.001 {
			at = point
			continue
		}
		for traveled := 0.0; traveled < distance; traveled += 20 {
			origin := quake.Vec3{
				at[0] + (point[0]-at[0])*traveled/distance,
				at[1] + (point[1]-at[1])*traveled/distance,
				at[2] + (point[2]-at[2])*traveled/distance,
			}
			if p.World.Geometry.DoorMoveHazard(movers, origin, point[0]-at[0], point[1]-at[1]) != "" {
				return false
			}
		}
		at = point
	}
	return true
}
