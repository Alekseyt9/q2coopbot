package bot

import (
	"fmt"
	"math"
	"q2coopbot/internal/quake"
)

const grenadeVelocityNormCap = 2000.0

type GrenadeBounceRange struct {
	Tick int        `json:"tick"`
	Min  quake.Vec3 `json:"min"`
	Max  quake.Vec3 `json:"max"`
}

type GrenadeBounceEnvelope struct {
	Scope             string               `json:"scope"`
	SpeedNormCap      float64              `json:"model_speed_norm_cap"`
	Gravity           float64              `json:"gravity"`
	Authorized        bool                 `json:"authorized"`
	GeometryCertified bool                 `json:"geometry_certified"`
	Compared          int                  `json:"compared,omitempty"`
	Ranges            []GrenadeBounceRange `json:"ranges"`
}

// Every native-style tick limits the speed norm before gravity and sweeping.
// Axis displacement is bounded by (speed cap + gravity*100ms)*100ms. Whatever the
// collision fraction/normal or settling decision, that endpoint lies within
// the sweep. Include bounce and no-bounce branches, and settled trajectories.
// This coarse bound is conditional on the model cap, not a server speed claim.
func grenadeBounceReach(seed grenadeBox, startTick, endTick int, gravity float64) *GrenadeBounceEnvelope {
	if startTick < 0 || endTick < startTick || endTick-startTick > 32 || gravity <= 0 || gravity > 2000 || math.IsNaN(gravity) || math.IsInf(gravity, 0) {
		return nil
	}
	for i := range seed.lo {
		if math.IsNaN(seed.lo[i]) || math.IsNaN(seed.hi[i]) || math.IsInf(seed.lo[i], 0) || math.IsInf(seed.hi[i], 0) || seed.lo[i] > seed.hi[i] {
			return nil
		}
	}
	r := &GrenadeBounceEnvelope{Scope: "coarse_post_collision_model_bound", SpeedNormCap: grenadeVelocityNormCap, Gravity: gravity}
	for tick := startTick; tick <= endTick; tick++ {
		distance := float64(tick-startTick) * .1 * (grenadeVelocityNormCap + gravity*.1)
		b := GrenadeBounceRange{Tick: tick}
		for i := range b.Min {
			b.Min[i] = seed.lo[i] - distance
			b.Max[i] = seed.hi[i] + distance
		}
		r.Ranges = append(r.Ranges, b)
	}
	return r
}

// Native positions remain held out. Seed from the predicted first bounce
// segment, never its observed endpoint, then check all later observations.
func checkGrenadeBounceReach(c *GrenadeCalibration, gravity float64) error {
	first := -1
	for i, p := range c.Points {
		if p.Bounces > 0 {
			first = i
			break
		}
	}
	if first < 0 || c.Launch == nil {
		return fmt.Errorf("bounce envelope input missing")
	}
	point := c.Points[first]
	previous := point.Predicted
	if first > 0 {
		previous = c.Points[first-1].Predicted
	}
	seed := grenadeBox{}
	for i := range seed.lo {
		seed.lo[i] = math.Min(previous[i], point.Predicted[i]) - 4
		seed.hi[i] = math.Max(previous[i], point.Predicted[i]) + 4
	}
	// Four units are the existing path calibration tolerance, not a refit.
	c.BounceEnvelope = grenadeBounceReach(seed, point.Frame, c.Launch.ExpectedExplosionFrame, gravity)
	if c.BounceEnvelope == nil || gravity <= 0 {
		return fmt.Errorf("invalid bounce envelope")
	}
	for _, p := range c.Points[first:] {
		index := p.Frame - point.Frame
		if index < 0 || index >= len(c.BounceEnvelope.Ranges) {
			return fmt.Errorf("post-bounce frame outside envelope")
		}
		b := c.BounceEnvelope.Ranges[index]
		for i := range p.Observed {
			if p.Observed[i] < b.Min[i] || p.Observed[i] > b.Max[i] {
				return fmt.Errorf("native post-bounce point outside model envelope")
			}
		}
		c.BounceEnvelope.Compared++
	}
	if c.BounceEnvelope.Compared < 3 {
		return fmt.Errorf("insufficient post-bounce observations")
	}
	return nil
}
