package learningenv

import (
	"math"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"testing"
)

func aimFixture() (RewardConfig, Step, ServerOutcome) {
	c, s, o := rewardFixture()
	c.Version = AimRewardVersion
	c.MonsterKill = 5
	c.AimPotential = .5
	c.AimGamma = .99
	solid := uint16(8290)
	clear := true
	s.Observation.Enemies = []policy.Enemy{{Distance: 100, Relative: quake.Vec3{100, 0, 0}, Solid: &solid, ClearShot: &clear}}
	s.Next.Enemies = s.Observation.Enemies
	return c, s, o
}

func TestAimPotentialDenseSignalAndTerminalBoundary(t *testing.T) {
	c, s, o := aimFixture()
	s.Observation.ViewAngles[1] = 16384
	r := c.Evaluate(&s, o)
	if !r.Available || math.Abs(r.Components["aim_potential"]-.25) > 1e-9 {
		t.Fatal(r)
	}
	s.Next.ViewAngles = s.Observation.ViewAngles
	r = c.Evaluate(&s, o)
	if math.Abs(r.Components["aim_potential"]-.0025) > 1e-9 {
		t.Fatal("discounted static potential", r)
	}
	s.Truncated = true
	s.Reason = "control_handoff"
	r = c.Evaluate(&s, o)
	if !r.Available || math.Abs(r.Components["aim_potential"]-.25) > 1e-9 {
		t.Fatal("handoff zero terminal potential", r)
	}
	s.Truncated = false
	s.Terminal = true
	s.Next.Health = 0
	o.Deaths = 1
	r = c.Evaluate(&s, o)
	if !r.Available || r.Components["death"] != -5 || math.Abs(r.Components["aim_potential"]-.25) > 1e-9 {
		t.Fatal("death evidence/potential", r)
	}
}

func TestAimPotentialDiscountedLoopCannotFarmReturn(t *testing.T) {
	c, s, o := aimFixture()
	// Equal start/end observation, then absorbing segment boundary.
	angles := []int16{0, 16384, -32768, 16384, 0}
	total := 0.0
	factor := 1.0
	for i := 0; i < len(angles)-1; i++ {
		s.Observation.ViewAngles[1] = angles[i]
		s.Next.ViewAngles[1] = angles[i+1]
		r := c.Evaluate(&s, o)
		if !r.Available {
			t.Fatal(r)
		}
		total += factor * r.Components["aim_potential"]
		factor *= c.AimGamma
	}
	if math.Abs(total) > 1e-9 {
		t.Fatal("nonzero discounted closed-loop shaping", total)
	}
	// General telescoping with nonzero initial/end potential.
	s.Observation.ViewAngles[1] = 16384
	s.Next.ViewAngles[1] = -32768
	a, _ := aimPotential(s.Observation, c.AimPotential)
	b, _ := aimPotential(*s.Next, c.AimPotential)
	r := c.Evaluate(&s, o)
	if math.Abs(r.Components["aim_potential"]-(c.AimGamma*b-a)) > 1e-9 {
		t.Fatal(r)
	}
}

func TestAimRewardUnknownBoundsAndInvalidConfig(t *testing.T) {
	c, s, o := aimFixture()
	s.Observation.Enemies[0].Solid = nil
	r := c.Evaluate(&s, o)
	if !r.Available || r.Components["aim_potential"] != 0 {
		t.Fatal("unknown bounds", r)
	}
	for _, v := range []float64{0, -1, 2, math.NaN(), math.Inf(1)} {
		bad := c
		bad.AimPotential = v
		if bad.Validate() == nil {
			t.Fatal("invalid scale", v)
		}
	}
	for _, v := range []float64{0, -1, 1, math.NaN()} {
		bad := c
		bad.AimGamma = v
		if bad.Validate() == nil {
			t.Fatal("invalid gamma", v)
		}
	}
	c.Version = KillRewardVersion
	if c.Validate() == nil {
		t.Fatal("v2 accepted shaping")
	}
}
