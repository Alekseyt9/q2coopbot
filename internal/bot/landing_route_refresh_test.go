package bot

import (
	"testing"

	"q2coopbot/internal/quake"
)

func TestFailedAirborneRouteInvalidatedOnLanding(t *testing.T) {
	goal := quake.Vec3{200, 0, 0}
	for _, tc := range []struct {
		name           string
		previousGround bool
		currentGround  bool
		wantKnown      bool
	}{
		{name: "landing", previousGround: false, currentGround: true, wantKnown: false},
		{name: "still_airborne", previousGround: false, currentGround: false, wantKnown: true},
		{name: "already_grounded", previousGround: true, currentGround: true, wantKnown: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := &Planner{
				World:      World{Map: "test", Geometry: &quake.MapInfo{}, Goal: "follow_teammate", Snapshot: quake.Snapshot{Map: "test", Frame: 10, Self: quake.Vec3{}, Teammate: &goal, Health: 100, OnGround: tc.previousGround}},
				routeKnown: true,
				routeOK:    false,
			}
			p.update(quake.Snapshot{Map: "test", Frame: 11, Self: quake.Vec3{}, Teammate: &goal, Health: 100, OnGround: tc.currentGround}, "")
			if p.routeKnown != tc.wantKnown {
				t.Fatalf("routeKnown=%v, want %v", p.routeKnown, tc.wantKnown)
			}
		})
	}
}
