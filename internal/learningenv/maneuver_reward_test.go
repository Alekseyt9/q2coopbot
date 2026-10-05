package learningenv

import (
	"math"
	"q2coopbot/internal/policy"
	"testing"
)

func maneuverFixture() (RewardConfig, Step, ServerOutcome) {
	c, s, o := aimFixture()
	c.Version = ManeuverRewardVersion
	c.SpacingPotential = 2
	c.ParasiteRange = 288
	s.Observation.Enemies[0].Class = "monster_parasite"
	s.Next.Enemies = append([]policy.Enemy{}, s.Observation.Enemies...)
	return c, s, o
}

func TestSpacingPotentialRetreatApproachAndVisibility(t *testing.T) {
	c, s, o := maneuverFixture()
	s.Observation.Enemies[0].Relative[0] = 144
	s.Next.Enemies[0].Relative[0] = 288
	r := c.Evaluate(&s, o)
	if !r.Available || math.Abs(r.Components["spacing_potential"]-1) > 1e-9 {
		t.Fatal("retreat", r)
	}
	s.Observation.Enemies[0].Relative[0] = 288
	s.Next.Enemies[0].Relative[0] = 144
	r = c.Evaluate(&s, o)
	if !r.Available || math.Abs(r.Components["spacing_potential"]+.99) > 1e-9 {
		t.Fatal("approach", r)
	}
	s.Observation.Enemies[0].Class = "monster_gunner"
	s.Next.Enemies[0].Class = "monster_gunner"
	r = c.Evaluate(&s, o)
	if r.Components["spacing_potential"] != 0 {
		t.Fatal("unrelated enemy", r)
	}
}

func TestSpacingPotentialLoopDeathAndConfig(t *testing.T) {
	c, s, o := maneuverFixture()
	distances := []float64{300, 144, 72, 144, 300}
	total, factor := 0.0, 1.0
	for i := 0; i < len(distances)-1; i++ {
		s.Observation.Enemies[0].Relative[0] = distances[i]
		s.Next.Enemies[0].Relative[0] = distances[i+1]
		r := c.Evaluate(&s, o)
		if !r.Available {
			t.Fatal(r)
		}
		total += factor * r.Components["spacing_potential"]
		factor *= c.AimGamma
	}
	if math.Abs(total) > 1e-9 {
		t.Fatal("range cycle farmed return", total)
	}
	s.Observation.Enemies[0].Relative[0] = 144
	s.Next.Health = 0
	s.Terminal = true
	o.Deaths = 1
	r := c.Evaluate(&s, o)
	if !r.Available || math.Abs(r.Components["spacing_potential"]-1) > 1e-9 || r.Components["death"] != -5 {
		t.Fatal(r)
	}
	for _, v := range []float64{0, 3, math.NaN()} {
		bad := c
		bad.SpacingPotential = v
		if bad.Validate() == nil {
			t.Fatal("scale", v)
		}
	}
	for _, v := range []float64{0, 279, 513, math.NaN()} {
		bad := c
		bad.ParasiteRange = v
		if bad.Validate() == nil {
			t.Fatal("range", v)
		}
	}
	c.Version = AimRewardVersion
	if c.Validate() == nil {
		t.Fatal("v3 accepted spacing fields")
	}
}
