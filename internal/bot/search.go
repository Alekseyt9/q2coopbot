package bot

import (
	"math"

	"q2coopbot/internal/quake"
)

// SearchAttempt records one bounded test of a viewpoint near the last sighting.
// It never asserts that the teammate is at the target.
type SearchAttempt struct {
	Entity              int         `json:"entity"`
	LastSeenFrame       int         `json:"last_seen_frame"`
	Target              *quake.Vec3 `json:"target,omitempty"`
	Basis               string      `json:"basis"`
	ExpectedObservation string      `json:"expected_observation"`
	Attempt             int         `json:"attempt"`
	MaxAttempts         int         `json:"max_attempts"`
	StartFrame          int         `json:"start_frame"`
	EndFrame            int         `json:"end_frame,omitempty"`
	State               string      `json:"state"`
	Outcome             string      `json:"outcome,omitempty"`
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
	if !p.searchApproachStarted {
		return quake.Vec3{}, "", false
	}
	if !p.probeAttempted {
		p.probeAttempted = true
		if viewpoint, ok := p.selectSearchViewpoint(s); ok {
			p.searchAttempt = &SearchAttempt{Entity: s.LastTeammateEntity,
				LastSeenFrame: s.Frame - *s.TeammateAgeFrames, Target: &viewpoint,
				Basis: "last_seen_aas_viewpoint", ExpectedObservation: "teammate_visible_in_current_snapshot",
				Attempt: 1, MaxAttempts: 1, StartFrame: s.Frame, State: "active"}
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
	travel, at := 0.0, from
	for _, waypoint := range route {
		if waypoint.Jump || waypoint.Kind == 11 || waypoint.ElevatorPhase != "" {
			return 0, false
		}
		travel += quake.Horizontal(at, waypoint.Position)
		at = waypoint.Position
	}
	travel += quake.Horizontal(at, goal)
	return travel, travel <= maxTravel
}

func (p *Planner) selectSearchViewpoint(s quake.Snapshot) (quake.Vec3, bool) {
	if p.Nav == nil || p.World.Geometry == nil || s.LastTeammate == nil {
		return quake.Vec3{}, false
	}
	last := *s.LastTeammate
	best := math.Inf(1)
	var chosen quake.Vec3
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
		route, ok := p.Nav.Route(s.Self, candidate)
		if !ok {
			continue
		}
		travel, safe := safeSearchRoute(route, s.Self, candidate, 320)
		if !safe || !p.searchRouteDoorsClear(s.Movers, s.Self, route, candidate) {
			continue
		}
		// Prefer a nearby different area without moving arbitrarily far from
		// the last observation. Ties remain stable in AAS area order.
		score := travel + fromLast*0.5
		if score < best {
			best, chosen = score, candidate
		}
	}
	return chosen, !math.IsInf(best, 1)
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
