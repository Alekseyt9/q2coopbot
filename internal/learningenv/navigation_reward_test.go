package learningenv

import (
	"math"
	"testing"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

func navigationFixture() (RewardConfig, Step) {
	c, s, _ := qualityFixture()
	c.Version, c.NavigationPotential = NavigationRewardVersion, .1
	goal := quake.Vec3{512, 0, 0}
	s.Observation.Position = quake.Vec3{}
	s.Next.Position = quake.Vec3{32, 0, 0}
	s.Observation.Teammate = &goal
	s.Observation.Navigation = &policy.NavigationContext{Goal: "follow_teammate", Status: "ready", GoalRelative: &goal}
	return c, s
}

func TestNavigationRewardOwnProgress(t *testing.T) {
	c, s := navigationFixture()
	r, e, err := c.navigationShaping(&s, false)
	if err != nil || r <= 0 || !e.Available || e.Before != 512 || e.After != 480 {
		t.Fatal(r, e, err)
	}
	s.Next.Position = s.Observation.Position
	near := quake.Vec3{1, 0, 0}
	s.Next.Teammate = &near
	s.Next.Navigation = &policy.NavigationContext{Goal: "follow_teammate", Status: "ready", GoalRelative: &near}
	r, e, err = c.navigationShaping(&s, false)
	if err != nil || r > 0 || e.Before != e.After {
		t.Fatal("moving partner rewarded stationary actor", r, e, err)
	}
}

func TestNavigationRewardDiscountedLoop(t *testing.T) {
	c, s := navigationFixture()
	positions := []quake.Vec3{{0, 0, 0}, {32, 0, 0}, {32, 32, 0}, {0, 32, 0}, {0, 0, 0}}
	total, discount := 0.0, 1.0
	for i := 0; i < len(positions)-1; i++ {
		s.Observation.Position, s.Next.Position = positions[i], positions[i+1]
		relative := quake.Vec3{512 - positions[i][0], -positions[i][1], 0}
		s.Observation.Navigation.GoalRelative = &relative
		r, _, err := c.navigationShaping(&s, false)
		if err != nil {
			t.Fatal(err)
		}
		total += discount * r
		discount *= c.AimGamma
	}
	if total > 1e-12 {
		t.Fatal("closed route rewarded", total)
	}
}

func TestNavigationRewardMasksAndTerminal(t *testing.T) {
	for _, scenario := range []string{"missing", "unreachable", "unsupported", "unseen", "stale", "teleport"} {
		t.Run(scenario, func(t *testing.T) {
			c, s := navigationFixture()
			n := s.Observation.Navigation
			switch scenario {
			case "missing":
				s.Observation.Navigation = nil
			case "unreachable":
				n.Status = "unreachable"
			case "unsupported":
				n.Goal = "collect_item"
			case "unseen":
				s.Observation.Teammate = nil
			case "stale":
				age := 201
				n.Goal, n.LastTeammateAgeFrames = "search_last_seen", &age
			case "teleport":
				s.Next.Position[0] = 129
			}
			r, e, err := c.navigationShaping(&s, false)
			if err != nil || r != 0 || e.Available || e.Reason == "" {
				t.Fatal(r, e, err)
			}
		})
	}
	c, s := navigationFixture()
	r, e, err := c.navigationShaping(&s, true)
	if err != nil || r >= 0 || e.AfterPotential != 0 {
		t.Fatal(r, e, err)
	}
	waypoint := quake.Vec3{64, 0, 0}
	s.Observation.Navigation.WaypointRelative = &waypoint
	_, e, err = c.navigationShaping(&s, false)
	if err != nil || e.Reference != "waypoint" || e.Before != 64 || e.After != 32 {
		t.Fatal(e, err)
	}
	s.Next.Position[0] = math.NaN()
	if _, _, err = c.navigationShaping(&s, false); err == nil {
		t.Fatal("nonfinite position accepted")
	}
}

func TestNavigationRewardConfigAndIntegration(t *testing.T) {
	c, s := navigationFixture()
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	_, _, outcome := qualityFixture()
	r := c.Evaluate(&s, outcome)
	if !r.Available || r.Navigation == nil || r.Components["navigation_potential"] <= 0 {
		t.Fatal(r)
	}
	for _, value := range []float64{0, -.1, .51, math.NaN(), math.Inf(1)} {
		bad := c
		bad.NavigationPotential = value
		if bad.Validate() == nil {
			t.Fatal("invalid coefficient accepted", value)
		}
	}
	c.Version = SelectedAimRewardVersion
	if c.Validate() == nil {
		t.Fatal("legacy reward accepted navigation coefficient")
	}
	c.NavigationPotential = 0
	r = c.Evaluate(&s, outcome)
	if !r.Available || r.Navigation != nil {
		t.Fatal("legacy reward changed", r)
	}
	if _, exists := r.Components["navigation_potential"]; exists {
		t.Fatal("legacy component changed")
	}
}
