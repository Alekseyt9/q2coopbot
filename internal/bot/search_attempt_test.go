package bot

import (
	"testing"

	"q2coopbot/internal/quake"
)

func searchAttemptFixture() (*Planner, quake.Snapshot, quake.Vec3) {
	last, view := quake.Vec3{200, 0, 24}, quake.Vec3{100, 0, 24}
	age := 3
	p := &Planner{Nav: &quake.Navigator{Areas: []quake.Area{{},
		{Min: quake.Vec3{80, -20, 0}, Max: quake.Vec3{170, 20, 48}}}},
		World: World{Map: "test", GeometryStatus: "ready"}, probeTarget: &view,
		probeAttempted: true, probeProgressFrame: 10, probeLastSelf: quake.Vec3{145, 0, 24},
		searchAttempt: &SearchAttempt{Entity: 2, LastSeenFrame: 8, Target: &view,
			Basis: "last_seen_aas_viewpoint", ExpectedObservation: "teammate_visible_in_current_snapshot",
			Attempt: 1, MaxAttempts: 1, StartFrame: 10, State: "active"}}
	s := quake.Snapshot{Map: "test", Frame: 11, Self: quake.Vec3{145, 0, 24},
		LastTeammate: &last, LastTeammateEntity: 2, TeammateAgeFrames: &age, Health: 100, OnGround: true}
	return p, s, view
}

func TestSearchAttemptEndsWithoutRepeatingWhenViewpointShowsNobody(t *testing.T) {
	p, s, view := searchAttemptFixture()
	s.Self = view
	p.update(s, "")
	if got := p.World.SearchAttempt; got == nil || got.State != "completed" || got.Outcome != "not_seen" ||
		got.EndFrame != 11 || got.MaxAttempts != 1 || p.World.Goal != "wait_for_teammate" {
		t.Fatalf("reached viewpoint did not close search: %+v", got)
	}
	s.Frame = 12
	age := 4
	s.TeammateAgeFrames = &age
	p.update(s, "")
	if p.World.Goal != "wait_for_teammate" || p.World.SearchAttempt == nil || p.World.SearchAttempt.Outcome != "not_seen" {
		t.Fatalf("completed probe restarted: %+v", p.World.SearchAttempt)
	}
}

func TestSearchAttemptEndsOnFreshSightOrUnsafeRoute(t *testing.T) {
	p, s, _ := searchAttemptFixture()
	p.update(s, "")
	if p.World.SearchAttempt == nil || p.World.SearchAttempt.State != "active" {
		t.Fatalf("probe did not become active: %+v", p.World.SearchAttempt)
	}
	visible := quake.Vec3{200, 0, 24}
	s.Frame, s.Teammate, s.TeammateEntity = 12, &visible, 2
	age := 0
	s.TeammateAgeFrames = &age
	p.update(s, "")
	if got := p.World.SearchAttempt; got == nil || got.Outcome != "reacquired" || got.EndFrame != 12 {
		t.Fatalf("fresh sighting did not close probe: %+v", got)
	}

	p, s, view := searchAttemptFixture()
	p.Nav = &quake.Navigator{Areas: []quake.Area{{},
		{Min: quake.Vec3{135, -10, 0}, Max: quake.Vec3{155, 10, 48}},
		{Min: quake.Vec3{90, -10, 0}, Max: quake.Vec3{110, 10, 48}}}, Edges: make([][]quake.Edge, 3)}
	p.probeTarget = &view
	p.update(s, "")
	if got := p.World.SearchAttempt; got == nil || got.Outcome != "route_unavailable" ||
		p.World.Goal != "wait_for_teammate" || p.hasGoal {
		t.Fatalf("unsafe probe route remained executable: %+v", got)
	}
}

func TestSearchBudgetRejectionDoesNotEraseCachedGraphRoute(t *testing.T) {
	p, s, _ := searchAttemptFixture()
	last := quake.Vec3{300, 0, 24}
	s.LastTeammate = &last
	p.probeTarget, p.searchAttempt = nil, nil
	p.probeAttempted = false
	p.GameClock = true
	p.Nav = &quake.Navigator{Areas: []quake.Area{{},
		{Min: quake.Vec3{130, -20, 0}, Max: quake.Vec3{160, 20, 48}},
		{Min: quake.Vec3{290, -20, 0}, Max: quake.Vec3{310, 20, 48}}},
		Edges: [][]quake.Edge{nil, {{To: 2, Kind: 2, Cost: 1, Start: quake.Vec3{1000, 0, 24}, End: last}}, nil}}
	for frame := 11; frame <= 12; frame++ {
		s.Frame = frame
		age := frame - 8
		s.TeammateAgeFrames = &age
		p.update(s, "")
		check := p.World.SearchRoute
		if check == nil || !check.GraphRouteFound || check.Reason != "distance_budget_exceeded" || check.Travel == nil || *check.Travel <= 640 || !p.routeOK {
			t.Fatalf("frame %d lost route/budget distinction: %+v", frame, check)
		}
		if p.World.Navigation != "unreachable" {
			t.Fatalf("budget bypassed: %s", p.World.Navigation)
		}
	}
}

func TestSearchBudgetDoesNotCountConsumedWaypointsAgain(t *testing.T) {
	p, s, _ := searchAttemptFixture()
	last := quake.Vec3{300, 0, 24}
	s.LastTeammate = &last
	p.probeTarget, p.searchAttempt = nil, nil
	p.probeAttempted = false
	p.GameClock = true
	p.World.Goal = "search_last_seen"
	p.routeKnown, p.routeOK = true, true
	p.target = last
	p.routeAt = p.navigationNow(s.Frame)
	p.route = []quake.Waypoint{{Position: quake.Vec3{-400, 0, 24}}, {Position: quake.Vec3{-300, 0, 24}},
		{Position: quake.Vec3{200, 0, 24}}, {Position: last}}
	p.routeIndex = 2
	p.update(s, "")
	check := p.World.SearchRoute
	if check == nil || check.Reason != "ready" || check.Travel == nil || *check.Travel != 155 || len(p.World.Route) != 2 {
		t.Fatalf("consumed path caused false budget refusal: %+v route=%v", check, p.World.Route)
	}
}

func TestSearchFallbackExplainsForbiddenRouteWithoutExecutingIt(t *testing.T) {
	p, s, _ := searchAttemptFixture()
	last := quake.Vec3{300, 0, 24}
	s.LastTeammate = &last
	p.probeTarget, p.searchAttempt = nil, nil
	p.probeAttempted = false
	p.Nav = &quake.Navigator{Areas: []quake.Area{{},
		{Min: quake.Vec3{130, -20, 0}, Max: quake.Vec3{160, 20, 48}},
		{Min: quake.Vec3{290, -20, 0}, Max: quake.Vec3{310, 20, 48}}},
		Edges: [][]quake.Edge{nil, {{To: 2, Kind: 4, Cost: 1, Start: s.Self, End: last}}, nil}}
	p.update(s, "")
	check := p.World.SearchRoute
	if check == nil || !check.GraphRouteFound || check.Reason != "requires_jump" || p.World.Navigation != "unreachable" || len(p.World.Route) != 0 {
		t.Fatalf("forbidden fallback became executable or lost reason: %+v", check)
	}
}

func TestApproachOnlyBaselineStopsBeforeViewpoint(t *testing.T) {
	p, s, _ := searchAttemptFixture()
	p.TestDisableProbe = true
	p.probeTarget, p.searchAttempt = nil, nil
	p.probeAttempted = false
	s.Self = quake.Vec3{100, 0, 24}
	if _, kind, ok := p.hiddenTeammateGoal(s); !ok || kind != "search_last_seen" {
		t.Fatalf("baseline lost approach: %s %v", kind, ok)
	}
	s.Self = quake.Vec3{150, 0, 24}
	if _, _, ok := p.hiddenTeammateGoal(s); ok || p.searchAttempt != nil || p.probeAttempted {
		t.Fatal("baseline started viewpoint after approach")
	}
	if !p.searchApproachStarted {
		t.Fatal("test never exercised approach phase")
	}
}
