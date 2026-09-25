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
