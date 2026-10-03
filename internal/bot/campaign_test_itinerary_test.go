package bot

import (
	"strings"
	"testing"

	"q2coopbot/internal/quake"
)

func TestCampaignTestItineraryAdvancesOnlyOnArrival(t *testing.T) {
	points := []quake.Vec3{{100, 0, 24}, {200, 0, 24}}
	p := &Planner{Campaign: true, testCampaignGoals: points, routeKnown: true}
	s := quake.Snapshot{Self: points[0], OnGround: false, Frame: 10}
	if goal, ok := p.campaignGoal(s); !ok || goal != points[0] || p.testCampaignGoalIndex != 0 {
		t.Fatal("airborne proximity advanced test itinerary")
	}
	s.OnGround = true
	p.campaignDependency = &CampaignDependency{State: "unit_activation_required", started: 1}
	if _, ok := p.campaignGoal(s); ok || p.testCampaignGoalIndex != 0 {
		t.Fatal("unresolved dependency advanced test itinerary")
	}
	p.campaignDependency = nil
	if goal, ok := p.campaignGoal(s); !ok || goal != points[1] || p.testCampaignGoalIndex != 1 || p.routeKnown {
		t.Fatal("arrival did not advance/invalidate previous route")
	}
	s.Self = points[1]
	if _, ok := p.campaignGoal(s); ok || p.World.Campaign.State != "test_waypoint_reached" || p.World.Campaign.CompletedLevels != 0 || p.World.Campaign.TestGoalIndex != 1 {
		t.Fatal("test itinerary mistaken for full campaign completion")
	}
}

func TestParseCampaignTestItinerary(t *testing.T) {
	points, err := parseTestCampaignGoals("1,2,3;4,5,6")
	if err != nil || len(points) != 2 || points[1] != (quake.Vec3{4, 5, 6}) {
		t.Fatal(points, err)
	}
	for _, value := range []string{"", "1,2,3;", "1,2,3;NaN,5,6", strings.Repeat("1,2,3;", 16) + "1,2,3"} {
		if _, err := parseTestCampaignGoals(value); err == nil {
			t.Fatal("invalid itinerary accepted", value)
		}
	}
}
